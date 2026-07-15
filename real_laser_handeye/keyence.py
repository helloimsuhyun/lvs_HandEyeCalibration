# -*- coding: 'unicode' -*-
try:
    from . import LJXAwrap
except ImportError:  # Direct execution: python real_laser_handeye/keyence.py
    import LJXAwrap
import math
import ctypes
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

image_available = False  # Flag to conrirm the completion of image acquisition.
image_time_ns = None      # Host time at which the SDK callback completed.
ysize_acquired = 0       # Number of Y lines of acquired image.
z_val = []               # The buffer for height image.
lumi_val = []            # The buffer for luminance image.


class Keyence:
    def __init__(
        self,
        line_width=5,
        ip_address=(192, 168, 1, 1),
        control_port=24691,
        high_speed_port=24692,
        device_id=0,
        timeout_sec=5.0,
    ):
        self.ysize = line_width     # Number of Y lines.
        self.ip_address = tuple(int(value) for value in ip_address)
        if len(self.ip_address) != 4 or any(not 0 <= value <= 255 for value in self.ip_address):
            raise ValueError("ip_address must contain four bytes")
        self.control_port = int(control_port)
        self.high_speed_port = int(high_speed_port)
        self.deviceId = int(device_id)
        self.timeout_sec = float(timeout_sec)
        self._opened = False

    def setup(self):
        global image_available
        global ysize_acquired
        global z_val
        global lumi_val

        use_external_batchStart = False     # 'True' if you start batch externally.

        ethernetConfig = LJXAwrap.LJX8IF_ETHERNET_CONFIG()
        for index, value in enumerate(self.ip_address):
            ethernetConfig.abyIpAddress[index] = value
        ethernetConfig.wPortNo = self.control_port

        ##################################################################
        # CHANGE THIS BLOCK TO MATCH YOUR SENSOR SETTINGS (TO HERE)
        ##################################################################

        # Ethernet open
        res = LJXAwrap.LJX8IF_EthernetOpen(self.deviceId, ethernetConfig)
        print("LJXAwrap.LJX8IF_EthernetOpen:", hex(res))
        if res != 0:
            raise ConnectionError(f"Keyence controller connection failed: {hex(res)}")
        self._opened = True

        # Initialize Hi-Speed Communication
        self.my_callback_s_a = LJXAwrap.LJX8IF_CALLBACK_SIMPLE_ARRAY(callback_s_a)

        res = LJXAwrap.LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray(
            self.deviceId,
            ethernetConfig,
            self.high_speed_port,
            self.my_callback_s_a,
            self.ysize,
            0)
        print("LJXAwrap.LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray:",
              hex(res))
        if res != 0:
            self.close()
            raise RuntimeError(f"Keyence high-speed initialization failed: {hex(res)}")

        # PreStart Hi-Speed Communication
        req = LJXAwrap.LJX8IF_HIGH_SPEED_PRE_START_REQ()
        req.bySendPosition = 2
        self.profinfo = LJXAwrap.LJX8IF_PROFILE_INFO()
        print(self.profinfo)

        res = LJXAwrap.LJX8IF_PreStartHighSpeedDataCommunication(
            self.deviceId,
            req,
            self.profinfo)
        print("LJXAwrap.LJX8IF_PreStartHighSpeedDataCommunication:", hex(res))
        if res != 0:
            self.close()
            raise RuntimeError(f"Keyence high-speed pre-start failed: {hex(res)}")

        # allocate the memory
        self.xsize = self.profinfo.wProfileDataCount
        z_val = [0] * self.xsize * self.ysize
        lumi_val = [0] * self.xsize * self.ysize

        # Start Hi-Speed Communication
        image_available = False
        res = LJXAwrap.LJX8IF_StartHighSpeedDataCommunication(self.deviceId)
        print("LJXAwrap.LJX8IF_StartHighSpeedDataCommunication:", hex(res))
        if res != 0:
            self.close()
            raise RuntimeError(f"Keyence high-speed start failed: {hex(res)}")

        # Start Measure (Start Batch)
        if use_external_batchStart is False:
            LJXAwrap.LJX8IF_StartMeasure(self.deviceId)

        # wait for the image acquisition complete
        start_time = time.time()
        while True:
            if image_available:
                break
            if time.time() - start_time > self.timeout_sec:
                break
            time.sleep(0.001)

        if image_available is not True:
            self.close()
            raise TimeoutError("Keyence profile acquisition timed out")

    def close(self):
        if not self._opened:
            return
        # stop measure
        LJXAwrap.LJX8IF_StopMeasure(self.deviceId)

        # Stop
        res = LJXAwrap.LJX8IF_StopHighSpeedDataCommunication(self.deviceId)
        print("LJXAwrap.LJX8IF_StoptHighSpeedDataCommunication:", hex(res))

        # Finalize
        res = LJXAwrap.LJX8IF_FinalizeHighSpeedDataCommunication(self.deviceId)
        print("LJXAwrap.LJX8IF_FinalizeHighSpeedDataCommunication:", hex(res))

        # Close
        res = LJXAwrap.LJX8IF_CommunicationClose(self.deviceId)
        print("LJXAwrap.LJX8IF_CommunicationClose:", hex(res))
        self._opened = False

    def live_view(self):
        global image_available
        global ysize_acquired
        global z_val
        global lumi_val
        fig = plt.figure()
        ##################################################################
        # Information of the acquired image
        ##################################################################
        cnt = 0
        
        try:
            while True:
                ZUnit = ctypes.c_ushort()
                LJXAwrap.LJX8IF_GetZUnitSimpleArray(self.deviceId, ZUnit)
                cnt += 1

                print("-----------------", cnt, "--------------------")
                print(" Luminance output      : ", self.profinfo.byLuminanceOutput)
                print(" Number of X points    : ", self.profinfo.wProfileDataCount)
                print(" Number of Y lines     : ", ysize_acquired)
                print(" X pitch in micrometer : ", self.profinfo.lXPitch / 100.0)
                print(" Z pitch in micrometer : ", ZUnit.value / 100.0)
                print("----------------------------------------")
                # Height image display
                sl = int(self.xsize * ysize_acquired / 2)  # the horizontal center profile

                x_val_mm = [0.0] * self.xsize
                z_val_mm = [0.0] * self.xsize
                for i in range(self.xsize):
                    # Conver X data to the actual length in millimeters
                    x_val_mm[i] = (self.profinfo.lXStart + self.profinfo.lXPitch * i)/100.0  # um
                    x_val_mm[i] /= 1000.0  # mm

                    # Conver Z data to the actual length in millimeters
                    if z_val[sl + i] == 0:  # invalid value
                        z_val_mm[i] = np.nan
                        x_val_mm[i] = np.nan
                    else:
                        # 'Simple array data' is offset to be unsigned 16-bit data.
                        # Decode by subtracting 32768 to get a signed value.
                        z_val_mm[i] = int(z_val[sl + i]) - 32768  # decode
                        z_val_mm[i] *= ZUnit.value / 100.0  # um
                        z_val_mm[i] /= 1000.0  # mm

                plotz_min = np.nanmin(z_val_mm)
                if np.isnan(plotz_min):
                    plotz_min = -1.0
                else:
                    plotz_min -= 1.0

                plotz_max = np.nanmax(z_val_mm)
                if np.isnan(plotz_max):
                    plotz_max = 1.0
                else:
                    plotz_max += 1.0

                plt.title("Gap Detection")

                z_val_mm = np.array([x for x in z_val_mm if math.isnan(x) == False])
                x_val_mm = np.array([x for x in x_val_mm if math.isnan(x) == False])
                plt.plot(x_val_mm, z_val_mm)
                plt.ylabel('Z-axis [mm]')
                plt.xlabel('X-axis [mm]')
                plt.draw()
                plt.pause(0.0001)
                plt.clf()
                image_available = False

        except KeyboardInterrupt:
            # Stop
            res = LJXAwrap.LJX8IF_StopHighSpeedDataCommunication(self.deviceId)
            print("LJXAwrap.LJX8IF_StoptHighSpeedDataCommunication:", hex(res))

            # Finalize
            res = LJXAwrap.LJX8IF_FinalizeHighSpeedDataCommunication(self.deviceId)
            print("LJXAwrap.LJX8IF_FinalizeHighSpeedDataCommunication:", hex(res))

            # Close
            res = LJXAwrap.LJX8IF_CommunicationClose(self.deviceId)
            print("LJXAwrap.LJX8IF_CommunicationClose:", hex(res))

            if image_available is not True:
                print("\nFailed to acquire image (timeout)")
                print("\nTerminated normally.")
                sys.exit()
            print("\nTerminated normally.")


# Callback function, It is called when the specified number of profiles are received.
def callback_s_a(p_header,
                 p_height,
                 p_lumi,
                 luminance_enable,
                 xpointnum,
                 profnum,
                 notify, user):

    global ysize_acquired
    global image_available
    global image_time_ns
    global z_val
    global lumi_val

    if (notify == 0) or (notify == 0x10000):
        if profnum != 0:
            if image_available is False:
                for i in range(xpointnum * profnum):
                    #print('i in callback', i)
                    z_val[i] = p_height[i]
                    if luminance_enable == 1:
                        lumi_val[i] = p_lumi[i]

                ysize_acquired = profnum
                image_time_ns = time.time_ns()
                image_available = True
    return


if __name__ == '__main__':
    K = Keyence()
    K.setup()
    K.live_view()
    K.close()
