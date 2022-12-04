/*******************************************************************************
* Copyright 2017 ROBOTIS CO., LTD.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/* Author: Ryu Woon Jung (Leon) */
// *********     Sync Write Example      *********
// Available Dynamixel model on this example : All models using Protocol 1.0
// This example is designed for using two Dynamixel MX-28, and an USB2DYNAMIXEL.
// To use another Dynamixel model, such as X series, see their details in E-Manual(emanual.robotis.com) and edit below "#define"d variables yourself.
// Be sure that Dynamixel MX properties are already set as %% ID : 1 / Baudnum : 34 (Baudrate : 57600)

// modified by sungryul lee, 11.16.2021
// dxl library header file for MX-12W, AX-12W, XC and XL model

#ifndef _DXL_H_
#define _DXL_H_

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#define STDIN_FILENO 0
#elif defined(_WIN32) || defined(_WIN64)
#include <conio.h>
#endif

#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"{
#endif
#include "dynamixel_sdk.h"                                 
#ifdef __cplusplus
}
#endif

// Control table address for AX-12W and MX-12W
#define ADDR_MX_TORQUE_ENABLE           24                  
#define ADDR_MX_GOAL_POSITION           30
#define ADDR_MX_PRESENT_POSITION        36
#define ADDR_MX_MOVING_SPEED            32

// Control table address for XL and XC model
#define ADDR_XL_TORQUE_ENABLE           64                 
#define ADDR_XL_OPERATING_MODE          11                 
#define ADDR_XL_GOAL_POSITION           116
#define ADDR_XL_PRESENT_POSITION        132
#define ADDR_XL_GOAL_VELOCITY           104
#define ADDR_XL_PRESENT_VELOCITY	128

// Data Byte Length for AX-12W and MX-12W
#define LEN_MX_GOAL_POSITION		2
#define LEN_MX_PRESENT_POSITION         2
#define LEN_MX_MOVING_SPEED             2

// Data Byte Length for XL and XC model
#define LEN_XL_GOAL_POSITION		4
#define LEN_XL_PRESENT_POSITION        	4
#define LEN_XL_GOAL_VELOCITY           	4
#define LEN_XL_PRESENT_VELOCITY        	4

// Protocol version for AX-12W and MX-12W
#define PROTOCOL_VERSION                1.0                 
// Protocol version for XL and XC model
//#define PROTOCOL_VERSION                2.0                 

// Default setting
#define DXL1_ID                         1		// Dynamixel#1 ID: 1
#define DXL2_ID                         2               // Dynamixel#2 ID: 2
#define BAUDRATE                        2000000
//#define BAUDRATE                        4000000
#define DEVICENAME                      "/dev/ttyUSB0"  // Check which port is being used on your controller
                                                        // ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"
#define TORQUE_ENABLE                   1               // Value for enabling the torque
#define TORQUE_DISABLE                  0               // Value for disabling the torque
#define OPMODE_XL_VELOCITY              1               
#define OPMODE_XL_POSITION              3               
#define OPMODE_XL_PWM	                16              

#define ESC_ASCII_VALUE                 0x1b

#ifdef __cplusplus
extern "C"{
#endif

int getch();
int kbhit(void);
unsigned int vel_convert(int speed);
int dxl_set_velocity(int goal_velocity1, int goal_velocity2);
void dxl_open();
void dxl_close();
void dxl_xl_open();
void dxl_xl_close();
int dxl_xl_syncwrite(int goal_velocity1, int goal_velocity2);

#ifdef __cplusplus
}
#endif

#endif //_DXL_H_

