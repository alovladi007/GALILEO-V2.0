#!/usr/bin/env python3
"""
Security Scanner
Performs automated security checks on the compliance module.
"""

import os
import sys
import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple


class SecurityIssue:
    """Represents a security issue found during scanning"""
    
    def __init__(self, severity: str, file: str, line: int, issue: str, recommendation: str):
        self.severity = severity  # CRITICAL, HIGH, MEDIUM, LOW, INFO
        self.file = file
        self.line = line
        self.issue = issue
        self.recommendation = recommendation
    
    def __str__(self):
        return f"[{self.severity}] {self.file}:{self.line} - {self.issue}\n  → {self.recommendation}"


class SecurityScanner:
    """Automated security scanner for Python code"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues: List[SecurityIssue] = []
    
    def scan_all(self) -> List[SecurityIssue]:
        """Run all security checks"""
        print("Starting security scan...\n")
        
        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            if "test" in str(file_path):
                continue  # Skip test files
            
            print(f"Scanning: {file_path.relative_to(self.project_root)}")
            self.scan_file(file_path)
        
        return self.issues
    
    def scan_file(self, file_path: Path):
        """Scan a single file for security issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Run various security checks
            self.check_hardcoded_secrets(file_path, lines)
            self.check_sql_injection(file_path, lines)
            self.check_insecure_functions(file_path, lines)
            self.check_weak_crypto(file_path, lines)
            self.check_input_validation(file_path, lines)
            self.check_error_handling(file_path, lines)
            self.check_logging_security(file_path, lines)
            
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
    
    def check_hardcoded_secrets(self, file_path: Path, lines: List[str]):
        """Check for hardcoded secrets"""
        patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Possible hardcoded password"),
            (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', "Possible hardcoded API key"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Possible hardcoded secret"),
            (r'token\s*=\s*["\'][^"\']+["\']', "Possible hardcoded token"),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, description in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Skip if it's a variable name or test data
                    if 'test' in line.lower() or 'example' in line.lower() or 'placeholder' in line.lower():
                        continue
                    
                    self.issues.append(SecurityIssue(
                        severity="HIGH",
                        file=str(file_path.relative_to(self.project_root)),
                        line=i,
                        issue=description,
                        recommendation="Use environment variables or secrets management system"
                    ))
    
    def check_sql_injection(self, file_path: Path, lines: List[str]):
        """Check for SQL injection vulnerabilities"""
        patterns = [
            r'execute\s*\([^)]*%[sd]',
            r'cursor\.execute\s*\([^)]*\+',
            r'\.format\s*\([^)]*\)\s*FROM',
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in patterns:
                if re.search(pattern, line):
                    self.issues.append(SecurityIssue(
                        severity="CRITICAL",
                        file=str(file_path.relative_to(self.project_root)),
                        line=i,
                        issue="Possible SQL injection vulnerability",
                        recommendation="Use parameterized queries with placeholders"
                    ))
    
    def check_insecure_functions(self, file_path: Path, lines: List[str]):
        """Check for use of insecure functions"""
        insecure = {
            'eval(': 'Avoid eval(), use ast.literal_eval() or safer alternatives',
            'exec(': 'Avoid exec(), use safer alternatives',
            'pickle.loads': 'Avoid pickle with untrusted data, use JSON or safer serialization',
            'yaml.load(': 'Use yaml.safe_load() instead of yaml.load()',
            'shell=True': 'Avoid shell=True in subprocess, use array form',
        }
        
        for i, line in enumerate(lines, 1):
            for func, recommendation in insecure.items():
                if func in line:
                    self.issues.append(SecurityIssue(
                        severity="HIGH",
                        file=str(file_path.relative_to(self.project_root)),
                        line=i,
                        issue=f"Use of insecure function: {func}",
                        recommendation=recommendation
                    ))
    
    def check_weak_crypto(self, file_path: Path, lines: List[str]):
        """Check for weak cryptographic practices"""
        weak_crypto = {
            'MD5': 'MD5 is cryptographically broken, use SHA-256 or stronger',
            'SHA1': 'SHA-1 is weak, use SHA-256 or stronger',
            'DES': 'DES is insecure, use AES',
            'random.random': 'Use secrets module for cryptographic randomness',
        }
        
        for i, line in enumerate(lines, 1):
            for crypto, recommendation in weak_crypto.items():
                if crypto in line and 'import' not in line.lower():
                    self.issues.append(SecurityIssue(
                        severity="MEDIUM",
                        file=str(file_path.relative_to(self.project_root)),
                        line=i,
                        issue=f"Weak cryptography: {crypto}",
                        recommendation=recommendation
                    ))
    
    def check_input_validation(self, file_path: Path, lines: List[str]):
        """Check for missing input validation"""
        # Look for functions that accept user input
        for i, line in enumerate(lines, 1):
            if 'input(' in line or 'request.' in line:
                # Check next few lines for validation
                has_validation = False
                for j in range(max(0, i-1), min(len(lines), i+5)):
                    if any(keyword in lines[j] for keyword in ['validate', 'check', 'sanitize', 'if ', 'assert']):
                        has_validation = True
                        break
                
                if not has_validation:
                    self.issues.append(SecurityIssue(
                        severity="MEDIUM",
                        file=str(file_path.relative_to(self.project_root)),
                        line=i,
                        issue="Possible missing input validation",
                        recommendation="Validate and sanitize all user inputs"
                    ))
    
    def check_error_handling(self, file_path: Path, lines: List[str]):
        """Check for poor error handling"""
        for i, line in enumerate(lines, 1):
            # Bare except clauses
            if re.match(r'\s*except\s*:', line):
                self.issues.append(SecurityIssue(
                    severity="LOW",
                    file=str(file_path.relative_to(self.project_root)),
                    line=i,
                    issue="Bare except clause",
                    recommendation="Catch specific exceptions instead of bare except"
                ))
            
            # Printing exceptions (info leak)
            if 'print(' in line and ('exception' in line.lower() or 'error' in line.lower()):
                self.issues.append(SecurityIssue(
                    severity="LOW",
                    file=str(file_path.relative_to(self.project_root)),
                    line=i,
                    issue="Printing exception details",
                    recommendation="Log errors securely, avoid exposing stack traces to users"
                ))
    
    def check_logging_security(self, file_path: Path, lines: List[str]):
        """Check for security issues in logging"""
        sensitive_keywords = ['password', 'secret', 'token', 'key', 'credential']
        
        for i, line in enumerate(lines, 1):
            if 'log' in line.lower() or 'print' in line:
                for keyword in sensitive_keywords:
                    if keyword in line.lower():
                        self.issues.append(SecurityIssue(
                            severity="MEDIUM",
                            file=str(file_path.relative_to(self.project_root)),
                            line=i,
                            issue=f"Possible logging of sensitive data: {keyword}",
                            recommendation="Avoid logging sensitive information, use redaction"
                        ))
    
    def generate_report(self) -> str:
        """Generate security scan report"""
        if not self.issues:
            return "\n✅ No security issues found!\n"
        
        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
        sorted_issues = sorted(self.issues, key=lambda x: severity_order.get(x.severity, 5))
        
        # Count by severity
        severity_counts = {}
        for issue in sorted_issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        
        # Generate report
        report = ["\n" + "="*80]
        report.append("SECURITY SCAN REPORT")
        report.append("="*80 + "\n")
        
        report.append("Summary:")
        report.append(f"  Total Issues: {len(sorted_issues)}")
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
            count = severity_counts.get(severity, 0)
            if count > 0:
                report.append(f"  {severity}: {count}")
        
        report.append("\n" + "-"*80 + "\n")
        report.append("Issues:\n")
        
        for issue in sorted_issues:
            report.append(str(issue))
            report.append("")
        
        report.append("="*80 + "\n")
        
        return "\n".join(report)


def run_security_scan(project_root: str = "/home/claude/project"):
    """Run security scan and display results"""
    scanner = SecurityScanner(project_root)
    issues = scanner.scan_all()
    report = scanner.generate_report()
    
    print(report)
    
    # Return exit code based on severity
    has_critical = any(issue.severity == "CRITICAL" for issue in issues)
    has_high = any(issue.severity == "HIGH" for issue in issues)
    
    if has_critical:
        return 2
    elif has_high:
        return 1
    else:
        return 0


if __name__ == "__main__":
    exit_code = run_security_scan()
    sys.exit(exit_code)
