# Visual to Tactile with Line-Us plotter

================================================

-   The app is: main.py
-   The buildozer file is: buildozer.spec
-   The Kivy file is: main.kv
-   The model files are: yolov3-tiny.cfg, yolov3-tiny.weights, coco.names
-   The shape database is: shapes_db.py
-   The logger can be configured in: main.py (search for "Logger")
-   The Line-us connection is in: main.py (search for "LineUs")

================================================

# How to build the app

## Install dependencies

```bash
pip install kivy opencv-python numpy
```

## Install buildozer

```bash
pip install buildozer
```

## Initialize buildozer

```bash
buildozer init
```

## Specify requirements (and other details) in buildozer.spec

```spec
# Change the title
title = Visual to Tactile with Line-Us plotter

# Change the package name
package.name = lineusdraw

# Change the package domain
package.domain = org.example

# Change the source directory
source.dir = .

# Change source include extensions
source.include_exts = py,png,jpg,kv,atlas,ttf,weights,cfg,names

# Change the requirements
requirements = python3,kivy,opencv,numpy
# Lately added for webcam: ffpyplayer
# requirements = python3,kivy,opencv,numpy,ffpyplayer


# Change app permissions
android.permissions = android.permission.CAMERA, android.permission.INTERNET
```

## Build the app

```bash
buildozer android debug deploy run
```

## Test the app

-   Connect to Line-us via WiFi
-   Ensure Line-us is on and connected to the same network
-   Don't forget to give permissions to the camera before running the app
-   Run the app
-   Point the camera to an object
-   The app should detect the object and draw it on the Line-us

---

---

---

## Actual changes

### ❗ Problems

-   The Windows environment couldn't build the app, required mannual installation of Android SDK and NDK
-   The Android device couldn't connect to Line-us, required to change the Line-us IP in main.py
-   The misorintation of the camera required to rotate the image in main.py

### ✅ Solution of Windows build environment

-   Using WSL to build the app as many people recommended for buildozer

#### How to install WSL

1. First thing is to enable WSL feature in Windows if not enabled already. To do this, open PowerShell as Administrator and run the following command:

    ```bash
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    ```

    then restart your pc.

2. Download and install Ubuntu from Microsoft Store.

    or by:

    ```bash
    wsl --install
    ```

3. Open Ubuntu and update the system:
    ```bash
    sudo apt update
    sudo apt upgrade
    ```

#### App requirements on WSL

1. (Optional) There were no user on the system, so I had to add one and use it to build the app. To do this, run the following command:

    ```bash
    sudo adduser shaker
    ```

    and login as the new user to build the app.

    ```bash
    su shaker
    ```

2. Install python3 and pip:

    ```bash
    sudo apt install -y python3 python3-pip python3-venv
    ```

3. Create a virtual environment:

    ```bash
    python3 -m venv venv
    ```

4. Activate the virtual environment:

    ```bash
    source .venv/bin/activate
    ```

5. Setting up Kivy and Buildozer within WSL:

    ```bash
    sudo apt install -y git zip unzip openjdk-17-jdk autoconf libtool pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 cmake libffi-dev libssl-dev
    ```

6. Install Cython and Virtualenv (User-specific):

    ```bash
    pip3 install --user --upgrade Cython==0.29.33 virtualenv
    ```

7. Install Kivy and Buildozer:

    ```bash
    pip3 install --user --upgrade kivy buildozer
    ```

8. I created a folder on the WSL in /home/shaker/FYP_WSL/ and copied the app files there.

9. Finally, I was able to build the app using buildozer.

    ```bash
    buildozer -v android debug
    ```

10. (Optional) Deploying the app on the device:
    ```bash
    buildozer android debug deploy run
    ```

### Error Debugger

#### On Android device

1. Goindg to developer options and enable USB debugging.
2. Connect the device to the computer via USB.

#### On the Windows PowerShell as admin

1. Install usb usbipd on powershell:
    ```bash
    winget install --interactive --exact dorssel.usbipd-win
    ```
2. Run the following command:
    ```bash
    usbipd list
    ```
3. Make the device shared:
    ```bash
    usbipd bind --busid <busid>
    ```
4. (May be required) Set default WSL distro:    
    ```bash
    wsl --setdefault <DistributionName>
    # or 
    # wsl -s <DistributionName>
    ```
5. Run the following command:
    ```bash
    usbipd attach -w --busid <busid> #--name <distro_name>
    ```
    where busid is the busid of the device and distro_name is the name of the distro(Ubuntu).

#### On the WSL

1. Installing the ADB:
    ```bash
    sudo apt install -y android-tools-adb
    ```
2. Check if the device is connected:
    ```bash
    adb devices
    ```
3. If the device is not connected, run the following command:
    ```bash
    adb kill-server
    adb start-server
    ```
4. (May be required) Mannually authorize the computer on the device.

    ```bash
    sudo nano /etc/udev/rules.d/51-android.rules
    ```

    and add the following line:

    ```bash
    SUBSYSTEM=="usb", ATTR{idVendor}=="vendor_id", ATTR{idProduct}=="your_product_id", MODE="0666", GROUP="plugdev"
    ```

    where vendor_id is the vendor id of the device and product_id is the product id of the device.

    then reload the udev rules:

    ```bash
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    ```

    then restart ADB server:

    ```bash
    adb kill-server
    adb start-server
    adb devices
    ```

#### Extra commands may be required

1. Updating and restarting the WSL:
    ```bash
    wsl --update
    wsl --shutdown
    ```
2. to debug the app that is running on the android device on the WSL using ADB:
    ```bash
    adb logcat -c  # Clear previous logs
    adb logcat python:D Kivy:D DEBUG:D AndroidRuntime:E *:S
    ```
