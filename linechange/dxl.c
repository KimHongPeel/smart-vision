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
// *********     Sync Write Example      *****cd ****
// Available Dynamixel model on this example : All models using Protocol 1.0
// This example is designed for using two Dynamixel MX-28, and an USB2DYNAMIXEL.
// To use another Dynamixel model, such as X series, see their details in E-Manual(emanual.robotis.com) and edit below "#define"d variables yourself.
// Be sure that Dynamixel MX properties are already set as %% ID : 1 / Baudnum : 34 (Baudrate : 57600)

// modified by sungryul lee, 11.16.2021
// dxl library source file for MX-12W and AX-12W

#include"dxl.h"

static int port_num;
static int group_num;
static int dxl_comm_result = COMM_TX_FAIL;             // Communication result
static uint8_t dxl_addparam_result = False;            // AddParam result
static uint8_t dxl_error = 0;                          // Dynamixel error

int getch()
{
#if defined(__linux__) || defined(__APPLE__)
  struct termios oldt, newt;
  int ch;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  ch = getchar();
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  return ch;
#elif defined(_WIN32) || defined(_WIN64)
  return _getch();
#endif
}

int kbhit(void)
{
#if defined(__linux__) || defined(__APPLE__)
  struct termios oldt, newt;
  int ch;
  int oldf;

  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

  ch = getchar();

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, oldf);

  if (ch != EOF)
  {
    ungetc(ch, stdin);
    return 1;
  }

  return 0;
#elif defined(_WIN32) || defined(_WIN64)
  return _kbhit();
#endif
}

// open port and enable dxl torque
void dxl_open()
{
  // Initialize PortHandler Structs
  // Set the port path
  // Get methods and members of PortHandlerLinux or PortHandlerWindows
  port_num = portHandler(DEVICENAME);

  // Initialize PacketHandler Structs
  packetHandler();

  // Initialize Groupsyncwrite instance
  //int group_num = groupSyncWrite(port_num, PROTOCOL_VERSION, ADDR_MX_GOAL_POSITION, LEN_MX_GOAL_POSITION);
  group_num = groupSyncWrite(port_num, PROTOCOL_VERSION, ADDR_MX_MOVING_SPEED, LEN_MX_GOAL_POSITION);

  // Open port
  if (openPort(port_num))
  {
    printf("Succeeded to open the port!\n");
  }
  else
  {
    printf("Failed to open the port!\n");
    printf("Press any key to terminate...\n");
    getch();
    exit(1);
  }

  // Set port baudrate
  if (setBaudRate(port_num, BAUDRATE))
  {
    printf("Succeeded to change the baudrate!\n");
  }
  else
  {
    printf("Failed to change the baudrate!\n");
    printf("Press any key to terminate...\n");
    getch();
    exit(1);
  }

  // Enable Dynamixel#1 Torque
  write1ByteTxRx(port_num, PROTOCOL_VERSION, DXL1_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE);
  if ((dxl_comm_result = getLastTxRxResult(port_num, PROTOCOL_VERSION)) != COMM_SUCCESS)
  {
    printf("%s\n", getTxRxResult(PROTOCOL_VERSION, dxl_comm_result));
    exit(1);
  }
  else if ((dxl_error = getLastRxPacketError(port_num, PROTOCOL_VERSION)) != 0)
  {
    printf("%s\n", getRxPacketError(PROTOCOL_VERSION, dxl_error));
    exit(1);
  }
  else
  {
    printf("Dynamixel#%d is ready\n", DXL1_ID);
  }

  // Enable Dynamixel#2 Torque
  write1ByteTxRx(port_num, PROTOCOL_VERSION, DXL2_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE);
  if ((dxl_comm_result = getLastTxRxResult(port_num, PROTOCOL_VERSION)) != COMM_SUCCESS)
  {
    printf("%s\n", getTxRxResult(PROTOCOL_VERSION, dxl_comm_result));
    exit(1);
  }
  else if ((dxl_error = getLastRxPacketError(port_num, PROTOCOL_VERSION)) != 0)
  {
    printf("%s\n", getRxPacketError(PROTOCOL_VERSION, dxl_error));
    exit(1);
  }
  else
  {
    printf("Dynamixel#%d is ready\n", DXL2_ID);
  }
}


// open port and enable dxl torque for xl model
void dxl_xl_open()
{
  // Initialize PortHandler Structs
  // Set the port path
  // Get methods and members of PortHandlerLinux or PortHandlerWindows
  port_num = portHandler(DEVICENAME);

  // Initialize PacketHandler Structs
  packetHandler();

  // Initialize Groupsyncwrite instance
  group_num = groupSyncWrite(port_num, PROTOCOL_VERSION, ADDR_XL_GOAL_VELOCITY, LEN_XL_GOAL_VELOCITY);

  // Open port
  if (openPort(port_num))
  {
    printf("Succeeded to open the port!\n");
  }
  else
  {
    printf("Failed to open the port!\n");
    printf("Press any key to terminate...\n");
    getch();
    exit(1);
  }

  // Set port baudrate
  if (setBaudRate(port_num, BAUDRATE))
  {
    printf("Succeeded to change the baudrate!\n");
  }
  else
  {
    printf("Failed to change the baudrate!\n");
    printf("Press any key to terminate...\n");
    getch();
    exit(1);
  }

  // set Dynamixel#1 to velocity mode
  write1ByteTxRx(port_num, PROTOCOL_VERSION, DXL1_ID, ADDR_XL_OPERATING_MODE, OPMODE_XL_VELOCITY);
  if ((dxl_comm_result = getLastTxRxResult(port_num, PROTOCOL_VERSION)) != COMM_SUCCESS)
  {
    printf("%s\n", getTxRxResult(PROTOCOL_VERSION, dxl_comm_result));
    printf("Failed to set operating mode!\n");
    printf("Press any key to terminate...\n");
    getch();
    exit(1);
  }
  else if ((dxl_error = getLastRxPacketError(port_num, PROTOCOL_VERSION)) != 0)
  {
    printf("%s\n", getRxPacketError(PROTOCOL_VERSION, dxl_error));
    printf("Failed to set operating mode!\n");
    printf("Press any key to terminate...\n");
    getch();
    exit(1);
  }
  else
  {
    printf("Dynamixel#%d has been successfully configured to velocity mode \n", DXL1_ID);
  }

  // set Dynamixel#2 to velocity mode
  write1ByteTxRx(port_num, PROTOCOL_VERSION, DXL2_ID, ADDR_XL_OPERATING_MODE, OPMODE_XL_VELOCITY);
  if ((dxl_comm_result = getLastTxRxResult(port_num, PROTOCOL_VERSION)) != COMM_SUCCESS)
  {
    printf("%s\n", getTxRxResult(PROTOCOL_VERSION, dxl_comm_result));
	printf("Failed to set operating mode!\n");
	printf("Press any key to terminate...\n");
    getch();
    exit(1);
  }
  else if ((dxl_error = getLastRxPacketError(port_num, PROTOCOL_VERSION)) != 0)
  {
    printf("%s\n", getRxPacketError(PROTOCOL_VERSION, dxl_error));
	printf("Failed to set operating mode!\n");
	printf("Press any key to terminate...\n");
    getch();
    exit(1);
  }
  else
  {
    printf("Dynamixel#%d has been successfully configured to velocity mode \n", DXL2_ID);
  }

  // Enable Dynamixel#1 Torque
  write1ByteTxRx(port_num, PROTOCOL_VERSION, DXL1_ID, ADDR_XL_TORQUE_ENABLE, TORQUE_ENABLE);
  if ((dxl_comm_result = getLastTxRxResult(port_num, PROTOCOL_VERSION)) != COMM_SUCCESS)
  {
    printf("%s\n", getTxRxResult(PROTOCOL_VERSION, dxl_comm_result));
	printf("Failed to set torque enable!\n");
	printf("Press any key to terminate...\n");
    getch();
    exit(1);
  }
  else if ((dxl_error = getLastRxPacketError(port_num, PROTOCOL_VERSION)) != 0)
  {
    printf("%s\n", getRxPacketError(PROTOCOL_VERSION, dxl_error));
	printf("Failed to set torque enable!\n");
	printf("Press any key to terminate...\n");
    getch();
    exit(1);
  }
  else
  {
    printf("Dynamixel#%d is ready\n", DXL1_ID);
  }

  // Enable Dynamixel#2 Torque
  write1ByteTxRx(port_num, PROTOCOL_VERSION, DXL2_ID, ADDR_XL_TORQUE_ENABLE, TORQUE_ENABLE);
  if ((dxl_comm_result = getLastTxRxResult(port_num, PROTOCOL_VERSION)) != COMM_SUCCESS)
  {
    printf("%s\n", getTxRxResult(PROTOCOL_VERSION, dxl_comm_result));
	printf("Failed to set torque enable!\n");
	printf("Press any key to terminate...\n");
    getch();
    exit(1);
  }
  else if ((dxl_error = getLastRxPacketError(port_num, PROTOCOL_VERSION)) != 0)
  {
    printf("%s\n", getRxPacketError(PROTOCOL_VERSION, dxl_error));
	printf("Failed to set torque enable!\n");
	printf("Press any key to terminate...\n");
    getch();
    exit(1);
  }
  else
  {
    printf("Dynamixel#%d is ready\n", DXL2_ID);
  }
}

// disable dxl torque and close port
void dxl_close()
{
  dxl_set_velocity(0,0);
 
  // Disable Dynamixel#1 Torque
  write1ByteTxRx(port_num, PROTOCOL_VERSION, DXL1_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE);
  if ((dxl_comm_result = getLastTxRxResult(port_num, PROTOCOL_VERSION)) != COMM_SUCCESS)
  {
    printf("%s\n", getTxRxResult(PROTOCOL_VERSION, dxl_comm_result));
  }
  else if ((dxl_error = getLastRxPacketError(port_num, PROTOCOL_VERSION)) != 0)
  {
    printf("%s\n", getRxPacketError(PROTOCOL_VERSION, dxl_error));
  }

  // Disable Dynamixel#2 Torque
  write1ByteTxRx(port_num, PROTOCOL_VERSION, DXL2_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE);
  if ((dxl_comm_result = getLastTxRxResult(port_num, PROTOCOL_VERSION)) != COMM_SUCCESS)
  {
    printf("%s\n", getTxRxResult(PROTOCOL_VERSION, dxl_comm_result));
  }
  else if ((dxl_error = getLastRxPacketError(port_num, PROTOCOL_VERSION)) != 0)
  {
    printf("%s\n", getRxPacketError(PROTOCOL_VERSION, dxl_error));
  }

  // Close port
  closePort(port_num);
}

// disable dxl torque and close port for xl model
void dxl_xl_close()
{
  dxl_xl_syncwrite(0,0);
  
  // Disable Dynamixel#1 Torque
  write1ByteTxRx(port_num, PROTOCOL_VERSION, DXL1_ID, ADDR_XL_TORQUE_ENABLE, TORQUE_DISABLE);
  if ((dxl_comm_result = getLastTxRxResult(port_num, PROTOCOL_VERSION)) != COMM_SUCCESS)
  {
    printf("%s\n", getTxRxResult(PROTOCOL_VERSION, dxl_comm_result));
  }
  else if ((dxl_error = getLastRxPacketError(port_num, PROTOCOL_VERSION)) != 0)
  {
    printf("%s\n", getRxPacketError(PROTOCOL_VERSION, dxl_error));
  }

  // Disable Dynamixel#2 Torque
  write1ByteTxRx(port_num, PROTOCOL_VERSION, DXL2_ID, ADDR_XL_TORQUE_ENABLE, TORQUE_DISABLE);
  if ((dxl_comm_result = getLastTxRxResult(port_num, PROTOCOL_VERSION)) != COMM_SUCCESS)
  {
    printf("%s\n", getTxRxResult(PROTOCOL_VERSION, dxl_comm_result));
  }
  else if ((dxl_error = getLastRxPacketError(port_num, PROTOCOL_VERSION)) != 0)
  {
    printf("%s\n", getRxPacketError(PROTOCOL_VERSION, dxl_error));
  }

  // Close port
  closePort(port_num);
}

//-1023 <= speed <= 1023 -> +(CCW) 0~1023, -(CW) 1024~2047
unsigned int vel_convert(int speed)
{
	unsigned int temp;
	if(speed > 1023) speed = 1023;
	else if (speed < -1023) speed = -1023;

	if(speed >= 0) temp = (unsigned int)speed;
	else temp = (unsigned int)(-speed + 1023);

        return temp;  
}

// goal_velocity1 : left motor speed, goal_velocity2 : right motor speed  
int dxl_set_velocity(int goal_velocity1, int goal_velocity2)
{
    // Add Dynamixel#1 goal position value to the Syncwrite storage
    dxl_addparam_result = groupSyncWriteAddParam(group_num, DXL1_ID, vel_convert(goal_velocity1), LEN_MX_GOAL_POSITION);
    if (dxl_addparam_result != True)
    {
      fprintf(stderr, "[ID:%03d] groupSyncWrite addparam failed", DXL1_ID);
      return 0;
    }

    // Add Dynamixel#2 goal position value to the Syncwrite parameter storage
    dxl_addparam_result = groupSyncWriteAddParam(group_num, DXL2_ID, vel_convert(goal_velocity2), LEN_MX_GOAL_POSITION);
    if (dxl_addparam_result != True)
    {
      fprintf(stderr, "[ID:%03d] groupSyncWrite addparam failed", DXL2_ID);
      return 0;
    }

    // Syncwrite goal velocity
    groupSyncWriteTxPacket(group_num);
    if ((dxl_comm_result = getLastTxRxResult(port_num, PROTOCOL_VERSION)) != COMM_SUCCESS)
      printf("%s\n", getTxRxResult(PROTOCOL_VERSION, dxl_comm_result));

    // Clear syncwrite parameter storage
    groupSyncWriteClearParam(group_num);
    return 0;
	
}

// for xl model
int dxl_xl_syncwrite(int goal_velocity1, int goal_velocity2)
{
    // Add Dynamixel#1 goal velocity value to the Syncwrite storage
    dxl_addparam_result = groupSyncWriteAddParam(group_num, DXL1_ID, goal_velocity1, LEN_XL_GOAL_VELOCITY);
    if (dxl_addparam_result != True)
    {
      fprintf(stderr, "[ID:%03d] groupSyncWrite addparam failed", DXL1_ID);
      return 0;
    }

    // Add Dynamixel#2 goal velocity value to the Syncwrite parameter storage
    dxl_addparam_result = groupSyncWriteAddParam(group_num, DXL2_ID, goal_velocity2, LEN_XL_GOAL_VELOCITY);
    if (dxl_addparam_result != True)
    {
      fprintf(stderr, "[ID:%03d] groupSyncWrite addparam failed", DXL2_ID);
      return 0;
    }

    // Syncwrite goal velocity
    groupSyncWriteTxPacket(group_num);
    if ((dxl_comm_result = getLastTxRxResult(port_num, PROTOCOL_VERSION)) != COMM_SUCCESS)
      printf("%s\n", getTxRxResult(PROTOCOL_VERSION, dxl_comm_result));

    // Clear syncwrite parameter storage
    groupSyncWriteClearParam(group_num);
    
    return 0;
	
}
