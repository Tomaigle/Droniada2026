#include <ESP32Servo.h>

Servo myServo;
int servoPin = 18;

void setup() {
  Serial.begin(115200);  // To talk to your Computer
  Serial2.begin(115200); // To talk to the Raspberry Pi

  myServo.setPeriodHertz(50);
  myServo.attach(servoPin, 500, 2400);

  Serial.println("ESP32 is ready. Waiting for Pi...");
}

void loop() {
  if (Serial2.available() > 0) {
    // Read the message from the Pi
    String msg = Serial2.readStringUntil('\n');
    int angle = msg.toInt(); // Convert text to number

    if (angle >= 0 && angle <= 180) {
      Serial.print("Moving to: ");
      Serial.println(angle);
      myServo.write(angle);
    }
  }
}
