// CMPE 491 - Smart Home Control System
// Emre Saygın & Talha Erden

const int LIGHT_PIN = 8;  // Relay or LED connected to pin 8
const int FAN_PIN   = 9;  // Relay or LED connected to pin 9

void setup() {
  Serial.begin(9600);  // Serial communication with Python

  pinMode(LIGHT_PIN, OUTPUT);
  pinMode(FAN_PIN, OUTPUT);

  // Initial state: all devices OFF
  digitalWrite(LIGHT_PIN, LOW);
  digitalWrite(FAN_PIN, LOW);
}

void loop() {
  // Check if data is available from serial port
  if (Serial.available() > 0) {
    char command = Serial.read();  // Read incoming command

    // --- EXECUTE COMMANDS ---
    if (command == 'L') {
      digitalWrite(LIGHT_PIN, HIGH);  // Turn light ON
    }
    else if (command == 'l') {
      digitalWrite(LIGHT_PIN, LOW);   // Turn light OFF
    }
    else if (command == 'F') {
      digitalWrite(FAN_PIN, HIGH);    // Turn fan ON
    }
    else if (command == 'f') {
      digitalWrite(FAN_PIN, LOW);     // Turn fan OFF
    }
  }
}
