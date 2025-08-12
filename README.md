# KRNL Key Automation Script

This Python script automates the process of retrieving KRNL keys and injecting them into the KRNL exploit.

## Features

- **Automatic Key Retrieval**: Automatically navigates to krnl.gg/getkey and retrieves a valid key
- **Key Validation**: Checks if a provided key matches the correct format
- **Clipboard Integration**: Automatically copies retrieved keys to clipboard
- **Automated Injection**: Uses pyautogui to automate the injection process

## Installation

1. Install Python 3.7+ if you haven't already
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the script:
   ```bash
   python krnl_automation.py
   ```

2. The script will ask if you already have a KRNL key:
   - If you have a valid key, enter it and the script will proceed with injection
   - If you don't have a key or enter an invalid one, the script will automatically retrieve one

## Important Notes

- **Coordinate Adjustment**: The `inject_krnl()` function uses hardcoded coordinates (500, 400) and (600, 500) for clicking. You'll need to adjust these coordinates based on your screen resolution and KRNL window position.
- **Browser Automation**: The script uses Chrome WebDriver to automate the key retrieval process.
- **Safety**: Make sure KRNL is the active window when the injection process starts.

## Dependencies

- `selenium`: Web automation
- `webdriver-manager`: Chrome driver management
- `pyautogui`: GUI automation
- `pyperclip`: Clipboard operations

## Disclaimer

This script is for educational purposes only. Use responsibly and in accordance with applicable terms of service.
