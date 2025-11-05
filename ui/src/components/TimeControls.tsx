'use client'

import { format } from 'date-fns'
import { 
  PlayIcon, 
  PauseIcon, 
  ForwardIcon, 
  BackwardIcon,
  ArrowPathIcon
} from '@heroicons/react/24/solid'

interface TimeControlsProps {
  selectedTime: Date
  timeRange: {
    start: Date
    end: Date
  }
  isPlaying: boolean
  playbackSpeed: number
  onTimeChange: (time: Date) => void
  onPlayPause: () => void
  onSpeedChange: (speed: number) => void
}

export function TimeControls({
  selectedTime,
  timeRange,
  isPlaying,
  playbackSpeed,
  onTimeChange,
  onPlayPause,
  onSpeedChange,
}: TimeControlsProps) {
  const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const percentage = parseFloat(event.target.value)
    const totalMs = timeRange.end.getTime() - timeRange.start.getTime()
    const newTimeMs = timeRange.start.getTime() + (totalMs * percentage / 100)
    onTimeChange(new Date(newTimeMs))
  }

  const currentPercentage = 
    ((selectedTime.getTime() - timeRange.start.getTime()) / 
     (timeRange.end.getTime() - timeRange.start.getTime())) * 100

  const speeds = [0.5, 1, 2, 5, 10]

  return (
    <div className="absolute bottom-4 left-4 right-4">
      <div className="glass-effect p-4 mx-auto max-w-3xl">
        {/* Time display */}
        <div className="flex justify-between items-center mb-3">
          <span className="text-sm text-gray-300">
            {format(timeRange.start, 'MMM dd, yyyy HH:mm')}
          </span>
          <span className="text-lg font-semibold text-white">
            {format(selectedTime, 'MMM dd, yyyy HH:mm:ss')}
          </span>
          <span className="text-sm text-gray-300">
            {format(timeRange.end, 'MMM dd, yyyy HH:mm')}
          </span>
        </div>

        {/* Time slider */}
        <div className="mb-4">
          <input
            type="range"
            min="0"
            max="100"
            step="0.01"
            value={currentPercentage}
            onChange={handleSliderChange}
            className="time-slider w-full"
          />
        </div>

        {/* Playback controls */}
        <div className="flex items-center justify-center space-x-4">
          <button
            onClick={() => onTimeChange(timeRange.start)}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            title="Reset to start"
          >
            <BackwardIcon className="h-5 w-5" />
          </button>

          <button
            onClick={onPlayPause}
            className="p-3 bg-primary-600 hover:bg-primary-700 rounded-full text-white transition-colors"
          >
            {isPlaying ? (
              <PauseIcon className="h-6 w-6" />
            ) : (
              <PlayIcon className="h-6 w-6" />
            )}
          </button>

          <button
            onClick={() => onTimeChange(timeRange.end)}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            title="Jump to end"
          >
            <ForwardIcon className="h-5 w-5" />
          </button>

          <div className="flex items-center space-x-2 ml-6">
            <ArrowPathIcon className="h-4 w-4 text-gray-400" />
            <select
              value={playbackSpeed}
              onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
              className="bg-slate-700 text-white text-sm rounded px-2 py-1 border border-slate-600"
            >
              {speeds.map(speed => (
                <option key={speed} value={speed}>
                  {speed}x
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
    </div>
  )
}
