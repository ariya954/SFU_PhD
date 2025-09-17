# Semantic-SplatBridge
This package implements a bridge between the [Robot Operating System](https://www.ros.org/) (ROS), and [Nerfstudio](https://docs.nerf.studio/en/latest/). This work builds upon [NerfBridge](https://github.com/javieryu/nerf_bridge), by focusing on [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) capability and embedding semantic information via [Open AI's CLIP Model](https://openai.com/index/clip/). 

<img src= real-time.gif height="300"> <img src= semantic.gif height="300">

## Requirements
- A Linux machine (tested with Ubuntu 22.04)
	- This should also have a CUDA capable GPU, and fairly strong CPU.
- ROS2 Humble installed on your Linux machine
- A camera that is compatible with ROS2 Humble
- Some means by which to estimate pose of the camera (SLAM, motion capture system, etc)

## Installation  
Below is a guide for installing Semantic-SplatBridge on an x86\_64 Ubuntu 22.04 machine with an NVIDIA GPU.

1. Install ROS2 Humble using the [installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html).

2. Install Miniconda (or Anaconda) using the [installation guide](https://docs.anaconda.com/free/miniconda/miniconda-install/).

3. Create a conda environment for Semantic-SplatBridge. Take note of the optional procedure for completely isolating the conda environment from your machine's base python site packages. For more details see this [StackOverflow post](https://stackoverflow.com/questions/25584276/how-to-disable-site-enable-user-site-for-an-environment) and [PEP-0370](https://peps.python.org/pep-0370/). 

    ```bash
    conda create --name splatbridge -y python=3.10

    conda activate splatbridge
    conda env config vars set PYTHONNOUSERSITE=1
    conda deactivate
    ```
4. Activate the conda environment and install Nerfstudio dependencies.
    ```bash
    # Activate conda environment, and upgrade pip
    conda activate splatbridge
    python -m pip install --upgrade pip

    # PyTorch, Torchvision dependency
    pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

    # CUDA dependency (by far the easiest way to manage cuda versions)
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

    # TinyCUDANN dependency (takes a while!)
    pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    ```
5. Clone, and install Semantic-SplatBridge.
    ```bash
    # Clone the Semantic-SplatBridge Repo
    git clone https://github.com/StanfordMSL/Semantic-SplatBridge.git

    # Install Semantic-SplatBridge (and Nerfstudio as a dependency)
    # Make sure your conda env is activated!
    cd Semantic-SplatBridge
    pip install -e . 
    ```
6. Install gsplat and ensure correct numpy version. 
    ```bash
    # Numpy 2.0 bricks the current install...
    pip install numpy==1.26.3

    # Uninstall the JIT version of Gsplat 1.0.0
    pip uninstall gsplat

    # Build GSPLAT 1.0.0
    pip install git+https://github.com/nerfstudio-project/gsplat.git@c7b0a383657307a13dff56cb2f832e3ab7f029fd

    pip install tyro==0.7.0

    # After this step you can never build GSplat again in the conda env so be careful!
    # Fix ROS2 CMake version dep
    conda install -c conda-forge gcc=12.1.0
    ```

Now you should be setup to run Semantic-SplatBridge. In the next section is a basic tutorial on training your first Gaussian Splat using Semantic-SplatBridge.

## Example using a ROSBag
This example simulates streaming data from a robot by replaying a [ROS2 bag](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html), and using Semantic-SplatBridge to train a Splatfacto model on that data. This example is a great way to check that your installation is working correctly.

These instructions are designed to be run in two different terminal sessions. They will be reffered to as terminals 1 and 2.

1. [Terminal 1] Download the example `wagon` rosbag using the provided download script. This script creates a folder `Semantic-SplatBridge/rosbags`, and downloads a rosbag to that directory.
    ```bash
    # Activate nerfbirdge conda env
    activate splatbridge

    # Run download script
    cd Semantic-SplatBridge
    python scripts/download_data.py
    ```

2. [Terminal 2] Start the `wagon` rosbag paused. The three important ROS topics in this rosbag are an image topic (`/zed/zed_node/rgb/image_rect_color/compressed`), a depth topic (`/zed/zed_node/depth/depth_registered`), and a camera pose topic (`/vrpn_mocap/fmu/pose`). These will be the data streams that will be used to train the Splatfacto model using Semantic-SplatBridge.

    ```bash
    # NOTE: for this example, Terminal 2 does not need to use the conda env.

    # Play the rosbag, but start it paused.
    cd Semantic-SplatBridge/rosbags
    ros2 bag play wagon --start-paused
    ```

3. [Terminal 1] Start Semantic-SplatBridge using the Nerfstudio `ns-train` CLI. Notice in the config file are parameters that can be changed from the base configuration. 

    ```bash
    ns-train ros-depth-splatfacto --data configs/zed_mocap.json
    ```

    After some initialization a message will appear stating that `(NerfBridge) Waiting to recieve 5 images...`, at this point you should open the Nerfstudio viewer in a browser tab to visualize the Splatfacto model as it trains. Training will start after completing the next step.

5. [Terminal 2] Press the SPACE key to start playing the rosbag. Once the pre-training image buffer is filled then training should commence, and the usual Nerfstudio print messages will appear in Terminal 1. After a few seconds the Nerfstudio viewer should start to show the recieved images as camera frames, and the Splatfacto model should begin be filled out.

6. After the rosbag in Terminal 2 finishes playing Semantic-SplatBridge will continue training the Splatfacto model on all of the data that it has recieved, but no new data will be added. You can use CTRL+c to kill Semantic-SplatBridge after you are done inspecting the Splatfacto model in the viewer.

## Running and Configuring Semantic-SplatBridge
The design and configuration of Semantic-SplatBridge is heavily inspired by Nerfstudio, and our recommendation is to become familiar with how that repository works before jumping into your own custom implementation of Semantic-SplatBridge.

Nerfstudio needs three key sources of data to train a NeRF: (1) color images, (2) camera poses corresponding to the color images, and (3) a camera model matching that of the camera that is being used. Semantic-SplatBridge expects that (1) and (2) are published to corresponding ROS image and pose topics, and that the names of these topics as well as (3) are provided in a JSON configuration file at launch. A sample Semantic-SplatBridge configuration JSON is provided at, `Semantic-SplatBridge/configs/zed_mocap.json` (this is the config used for the example above). We recommend using the [``camera_calibration``](http://wiki.ros.org/camera_calibration) package to determine the camera model parameters. 

To launch Semantic-SplatBridge we use the Nerfstudio CLI with the command below.
```
ns-train [METHOD-NAME] --data /path/to/config.json
```

## Our Setup
The following is a description of the setup that we at the Stanford Multi-robot Systems Lab have been using to train 3D Gaussian Splats online with Semantic-SplatBridge from images captured by a camera mounted to a quadrotor.

### Camera
We use a **Zed Mini** camera to provide our stream of images. This camera has quite a few features that make it really nice for working with 3D Gaussian Splats.

  - Integrated IMU: Vision only SLAM tends to be highly susceptible to visual artifacts and lighting conditions requiring more "babysitting". Nicely, the Zed Mini has an integrated IMU which can be used in conjunction with the camera to provide much more reliable pose estimates.
  - A Great ROS Package: The manufacturers of this camera provide documentation for how to set up ROS2 integration that is reasonably easy to work with, and is easy to install, [Zed ROS2](https://www.stereolabs.com/docs/ros2).

We currently use this camera mounted on a quadrotor that publishes images via WiFi to a ground station where the bulk of the computation takes place. 

**Alternatives:** Zed Mini cameras are expensive, and require more powerful computers to run. For a more economical choice see this camera from Arducam/UCTronics [OV9782](https://www.uctronics.com/arducam-global-shutter-color-usb-1mp-ov9782-uvc-webcam-module.html). This is a bare-bones USB 2.0 camera which can be used in conjunction with the [usb_cam](https://index.ros.org/r/usb_cam/) ROS2 Package.

### SLAM
The Zed ROS2 setup will automatically provide a pose topic `/zed/zed_node/pose` that estimates the pose of the camera. This approach results in fairly good 3D Gaussian Splat quality, especially in highly textured environments. However in environments with less texture, we found that there are some cases where drift occurs on the poses. For this reason, we use an example with motion capture poses in our ROS2 example.  

### Training Machine and On-board Computer
Our current computing setup is composed of two machines a training computer and an on-board computer which are connected via WiFi. The training computer is used to run Semantic-SplatBridge, and is a powerful workstation with an Intel(R) Core(TM) i9-13900K CPU, and an NVIDIA GeForce RTX 4090 GPU. The on-board computer is an NVIDIA Jetson Orin Nano DevKit directly mounted on a custom quadrotor, and is used to run the Zed Mini. At runtime, the on-board camera and training computer communicate over a local wireless network.

Alternatively, everything can run on a single machine with a camera, where the machine runs both the camera and the Semantic-SplatBridge training. Due to compute requirements this setup will likely not be very "mobile", but can be a good way to verify that everything is running smoothly before testing on robot hardware.

### Drone Workflow
In our typical workflow, we deploy the drone and start the Zed Mini on-board. Then, once the drone is in a steady hover we start Semantic-SplatBridge on the training machine, and begin the mapping flight orienting the camera towards areas of interest.

## Acknowledgements
Semantic-SplatBridge is entirely enabled by the first-class work of the [Nerfstudio Development Team and community](https://github.com/nerfstudio-project/nerfstudio/#contributors) and [NerfBridge](https://github.com/javieryu/nerf_bridge).

## Citation
In case anyone does use Semantic-SplatBridge as a starting point for any research please cite both the Nerfstudio and this repository.

```
# --------------------------- Nerfstudio -----------------------
@article{nerfstudio,
    author = {Tancik, Matthew and Weber, Ethan and Ng, Evonne and Li, Ruilong and Yi,
            Brent and Kerr, Justin and Wang, Terrance and Kristoffersen, Alexander and Austin,
            Jake and Salahi, Kamyar and Ahuja, Abhik and McAllister, David and Kanazawa, Angjoo},
    title = {Nerfstudio: A Modular Framework for Neural Radiance Field Development},
    journal = {arXiv preprint arXiv:2302.04264},
    year = {2023},
}


# --------------------------- NerfBridge ---------------------
@article{yu2023nerfbridge,
  title={NerfBridge: Bringing Real-time, Online Neural Radiance Field Training to Robotics},
  author={Yu, Javier and Low, Jun En and Nagami, Keiko and Schwager, Mac},
  journal={arXiv preprint arXiv:2305.09761},
  year={2023}
}


# --------------------------- Semantic-SplatBridge ---------------------
@article{nagami2025vista,
    title={VISTA: Open-Vocabulary, Task-Relevant Robot Exploration with Online Semantic Gaussian Splatting}, 
    author={Keiko Nagami and Timothy Chen and Javier Yu and Ola Shorinwa and Maximilian Adang and Carlyn Dougherty and Eric Cristofalo and Mac Schwager},
    journal={arXiv preprint arXiv:2507.01125}
    year={2025},
}
```
