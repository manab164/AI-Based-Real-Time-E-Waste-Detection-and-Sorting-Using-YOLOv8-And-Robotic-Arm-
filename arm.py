#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#define PCA_ADDR 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA_ADDR);
#define BASE_CH 0
#define SHOULDER_CH 1
#define ELBOW_CH 2
#define GRIPPER_CH 3
#define SERVOMIN 170 // ~0.83 ms
#define SERVOMAX 560 // ~2.73 ms
volatile bool stopRequested = false;
bool busy = false;
int currentAngle[4] = {90, 90, 90, 90};
const int DEFAULT_STEP_DELAY = 40;
int angleToPulse(int a){
a = constrain(a, 0, 180);
return map(a, 0, 180, SERVOMIN, SERVOMAX);
}
void setServoRaw(uint8_t ch, int angle){
angle = constrain(angle, 0, 180);
int p = angleToPulse(angle);
pwm.setPWM(ch, 0, p);
currentAngle[ch] = angle;
}
void setServo(uint8_t ch, int angle){
if (stopRequested) return;
setServoRaw(ch, angle);
}
void delayAbort(unsigned long ms){
unsigned long t0 = millis();
while (!stopRequested && millis() - t0 < ms){
delay(5);
}
}
void emergencyStop(){
stopRequested = true;
busy = false;
pwm.setPWM(BASE_CH, 0, 0);
pwm.setPWM(SHOULDER_CH, 0, 0);
pwm.setPWM(ELBOW_CH, 0, 0);
pwm.setPWM(GRIPPER_CH, 0, 0);
Serial.println("EMERGENCY STOP");
}
int PICK_BASE_ANG = 90;
int BIN1_BASE_ANG = 170;
int BIN2_BASE_ANG = 210;
int BIN3_BASE_ANG = 40;
int HOME_SHO=92, HOME_ELB=90;
int PICK_SHO=170, PICK_ELB=250;
int BIN_SHO =165, BIN_ELB =200
bool dropToBin(char code){
if (stopRequested) return false;
switch(code){
case 'L': goPoseRaw(BIN1_BASE_ANG, BIN_SHO, BIN_ELB); break;
case 'M': goPoseRaw(BIN2_BASE_ANG, BIN_SHO, BIN_ELB); break;
case 'H': goPoseRaw(BIN3_BASE_ANG, BIN_SHO, BIN_ELB); break;
default: return false;
}
if (stopRequested) return false;
openGripper();
delayAbort(200);
return !stopRequested;
}
unsigned long bootMs;
void setup(){
Serial.begin(9600);
Wire.begin();
Wire.setClock(400000);
pwm.begin();
pwm.setPWMFreq(50);
delay(200);
currentAngle[BASE_CH] = 90;
currentAngle[SHOULDER_CH] = HOME_SHO;
currentAngle[ELBOW_CH] = HOME_ELB;
currentAngle[GRIPPER_CH] = 90;
slowMoveChannel(BASE_CH, 80, 25); delayAbort(100);
slowMoveChannel(BASE_CH, 100, 25); delayAbort(100);
goHome();
openGripper();
bootMs = millis();
Serial.println("Arm ready.");
Serial.println("Commands: B### S### E### G### or L/M/H to sort, X to STOP.");
}
void loop(){
while (Serial.available()){
String s = Serial.readStringUntil('\n');
s.trim();
if (!s.length()) continue;
char c = toupper(s.charAt(0));
if (c == 'X'){
emergencyStop();
return;
}
if (stopRequested){
continue;
}
if (!busy){
if (c=='B' || c=='S' || c=='E' || c=='G'){
int val = s.substring(1).toInt();
val = constrain(val, 0, 180);
if (c=='B') slowMoveChannel(BASE_CH, val, DEFAULT_STEP_DELAY);
if (c=='S') slowMoveChannel(SHOULDER_CH, val, DEFAULT_STEP_DELAY);
if (c=='E') slowMoveChannel(ELBOW_CH, val, DEFAULT_STEP_DELAY);
if (c=='G') slowMoveChannel(GRIPPER_CH, val, DEFAULT_STEP_DELAY);
} else if (c=='L' || c=='M' || c=='H'){
busy = true;
if (pickSequence() && dropToBin(c) && !stopRequested){
goHome();
Serial.print("Done: "); Serial.println(c);
}
busy = false;
}
}
}
}