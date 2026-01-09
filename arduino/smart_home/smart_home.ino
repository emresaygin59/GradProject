/**
 * Sign Language Based Smart Home Control System
 * Arduino Firmware
 * * Functions:
 * - Reads single-character commands from Serial Port (USB).
 * - Controls Relay Module for Light and Fan.
 * - Updates LCD Display with status messages.
 */

#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// Initialize LCD. Check if your address is 0x27 or 0x3F.
LiquidCrystal_I2C lcd(0x27, 16, 2);

// Pin Definitions
const int PIN_LIGHT = 8; // Connected to Relay IN1 (Light)
const int PIN_FAN = 9;   // Connected to Relay IN2 (Fan)

void setup() {
  // Initialize Serial Communication
  Serial.begin(9600);

  // Set Pin Modes
  pinMode(PIN_LIGHT, OUTPUT);
  pinMode(PIN_FAN, OUTPUT);

  // Initialize LCD
  lcd.init();
  lcd.backlight();

  // Startup Message
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  delay(2000); // Keep startup message for 2 seconds
  lcd.clear();
}

void loop() {
  // Check if data is available on Serial Port
  if (Serial.available() > 0) {
    char incomingCommand = Serial.read();

    // Update screen only for valid commands
    if (incomingCommand == '1' || incomingCommand == '0' || incomingCommand == '3' || incomingCommand == '2') {

      lcd.clear(); // Clear previous text

      // --- LIGHT CONTROL ---
      if (incomingCommand == '1') {
        digitalWrite(PIN_LIGHT, HIGH);
        lcd.setCursor(0, 0);
        lcd.print("LIGHT STATUS:");
        lcd.setCursor(0, 1);
        lcd.print(">> ON");
      }
      else if (incomingCommand == '0') {
        digitalWrite(PIN_LIGHT, LOW);
        lcd.setCursor(0, 0);
        lcd.print("LIGHT STATUS:");
        lcd.setCursor(0, 1);
        lcd.print(">> OFF");
      }

      // --- FAN CONTROL ---
      else if (incomingCommand == '3') {
        digitalWrite(PIN_FAN, HIGH);
        lcd.setCursor(0, 0);
        lcd.print("FAN STATUS:");
        lcd.setCursor(0, 1);
        lcd.print(">> RUNNING");
      }
      else if (incomingCommand == '2') {
        digitalWrite(PIN_FAN, LOW);
        lcd.setCursor(0, 0);
        lcd.print("FAN STATUS:");
        lcd.setCursor(0, 1);
        lcd.print(">> STOPPED");
      }

      // --- USER FEEDBACK DELAY ---
      // Freeze execution for 3 seconds to keep the status message visible on the LCD.
      delay(3000);

      // Return to Idle State
      lcd.clear();
      lcd.print("Waiting Command");
    }
  }
}