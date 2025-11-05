import { test, expect } from '@playwright/test'

test.describe('Gravity Processing UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('should display the main page', async ({ page }) => {
    // Check if main elements are present
    await expect(page.locator('h1')).toContainText('Gravity Ops')
    await expect(page.locator('text=Dashboard')).toBeVisible()
    await expect(page.locator('text=Analytics')).toBeVisible()
    await expect(page.locator('text=Processing')).toBeVisible()
  })

  test('should show globe visualization', async ({ page }) => {
    // Wait for Cesium to load
    await page.waitForSelector('.cesium-viewer', { timeout: 10000 })
    
    // Check if globe controls are present
    await expect(page.locator('text=Visualization')).toBeVisible()
    await expect(page.locator('text=Gravity Field')).toBeVisible()
    await expect(page.locator('text=Satellites')).toBeVisible()
  })

  test('should toggle satellite visibility', async ({ page }) => {
    // Wait for satellite controls
    await page.waitForSelector('text=GRACE-A')
    
    // Toggle satellite
    const graceACheckbox = page.locator('input[type="checkbox"]').filter({ hasText: 'GRACE-A' })
    await graceACheckbox.uncheck()
    await expect(graceACheckbox).not.toBeChecked()
    
    await graceACheckbox.check()
    await expect(graceACheckbox).toBeChecked()
  })

  test('should show time controls', async ({ page }) => {
    // Check time controls are present
    await expect(page.locator('.time-slider')).toBeVisible()
    await expect(page.locator('button[title="Reset to start"]')).toBeVisible()
    await expect(page.locator('button[title="Jump to end"]')).toBeVisible()
  })

  test('should navigate between tabs in data panel', async ({ page }) => {
    // Click on Details tab
    await page.click('text=Details')
    await expect(page.locator('text=Satellite Details')).toBeVisible()
    
    // Click on Comparison tab
    await page.click('text=Comparison')
    await expect(page.locator('text=Run Comparison')).toBeVisible()
    
    // Go back to Overview
    await page.click('text=Overview')
    await expect(page.locator('text=Gravity Field Statistics')).toBeVisible()
  })

  test('should handle authentication flow', async ({ page }) => {
    // Click sign in
    await page.click('text=Sign In')
    
    // Should redirect to auth page
    await expect(page).toHaveURL(/.*auth/)
  })

  test('should be responsive', async ({ page }) => {
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 })
    
    // Check mobile menu button is visible
    await expect(page.locator('button[class*="md:hidden"]')).toBeVisible()
    
    // Open mobile menu
    await page.click('button[class*="md:hidden"]')
    
    // Check navigation items in mobile menu
    await expect(page.locator('text=Dashboard')).toBeVisible()
  })

  test('should show job console for authenticated users', async ({ page }) => {
    // Mock authentication by setting a cookie (in real test, would do proper auth)
    await page.context().addCookies([
      {
        name: 'next-auth.session-token',
        value: 'mock-session-token',
        domain: 'localhost',
        path: '/',
      }
    ])
    
    await page.reload()
    
    // Job console should be visible
    await expect(page.locator('text=Processing Jobs')).toBeVisible()
  })

  test('should display legend when gravity overlay is enabled', async ({ page }) => {
    // Ensure gravity overlay is checked
    const gravityCheckbox = page.locator('input[type="checkbox"]').filter({ hasText: 'Gravity Field' })
    await gravityCheckbox.check()
    
    // Legend should be visible
    await expect(page.locator('text=Gravity Anomaly (mGal)')).toBeVisible()
  })

  test('should handle playback controls', async ({ page }) => {
    // Find play button
    const playButton = page.locator('button').filter({ has: page.locator('svg[class*="h-6"]') }).first()
    
    // Click play
    await playButton.click()
    
    // Should show pause icon (button state changes)
    await expect(playButton).toBeVisible()
    
    // Click pause
    await playButton.click()
  })
})

test.describe('Accessibility', () => {
  test('should have proper ARIA labels', async ({ page }) => {
    await page.goto('/')
    
    // Check for proper heading hierarchy
    const h1 = await page.locator('h1').count()
    expect(h1).toBeGreaterThan(0)
    
    // Check for button accessibility
    const buttons = page.locator('button')
    const buttonCount = await buttons.count()
    
    for (let i = 0; i < buttonCount; i++) {
      const button = buttons.nth(i)
      const hasText = await button.textContent()
      const hasTitle = await button.getAttribute('title')
      const hasAria = await button.getAttribute('aria-label')
      
      // Button should have either text, title, or aria-label
      expect(hasText || hasTitle || hasAria).toBeTruthy()
    }
  })

  test('should be keyboard navigable', async ({ page }) => {
    await page.goto('/')
    
    // Tab through interactive elements
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
    
    // Should be able to activate with Enter
    await page.keyboard.press('Enter')
  })
})
