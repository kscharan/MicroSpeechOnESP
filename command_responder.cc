/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "command_responder.h"
#include "esp_now.h"
#include <cstring>
#include <stdio.h>

// The default implementation writes out the name of the recognized command
// to the error console. Real applications will want to take some custom
// action instead, and should implement their own versions of this function.
static uint8_t Lamp_mac_address[6] = {0x30, 0xAE, 0xA4, 0x97, 0xC1, 0xE8};
static uint8_t TV_mac_address[6] = {0x30, 0xAE, 0xA4, 0x97, 0xC1, 0xE9};
void RespondToCommand(tflite::ErrorReporter* error_reporter,
                      int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command,
                      int activateLampControl, int activateTvControl) {
  if (is_new_command) {

    bool commandDevice;           
    TF_LITE_REPORT_ERROR(error_reporter, "Heard %s (%d) @%dms", found_command,
                         score, current_time);
    if(strcmp(found_command, "on") == 0)
    {
      commandDevice = true;
      if(activateLampControl == 1)
      {
        esp_now_send(Lamp_mac_address, (const uint8_t *) &commandDevice, sizeof(commandDevice));
       TF_LITE_REPORT_ERROR(error_reporter, "switch on Lamp");
      }
      if(activateTvControl == 1)
      {
        esp_now_send(TV_mac_address, (const uint8_t *) &commandDevice, sizeof(commandDevice));
        TF_LITE_REPORT_ERROR(error_reporter, "switch on Tv");
      }
      
    }
    else if (strcmp(found_command, "off") == 0)
    {
      commandDevice = false;
      if(activateLampControl == 0)
      {
        esp_now_send(Lamp_mac_address, (const uint8_t *) &commandDevice, sizeof(commandDevice));
        TF_LITE_REPORT_ERROR(error_reporter, "switch off Lamp");
      }
      if(activateTvControl == 0)
      {
        esp_now_send(TV_mac_address, (const uint8_t *) &commandDevice, sizeof(commandDevice));
        TF_LITE_REPORT_ERROR(error_reporter, "switch off Tv");
      }
    }
    
  }
}
