Gaussian Splatting Reuse Guide on Jetson AGX Thor GPU
===========================

How to connect to the GPU
-------------------------
1. Connect the Thor board to power.

2. Connect the Ethernet cable between the Thor board and the laptop.

3. On the laptop, open:
   Settings

4. Go to:
   Network & Internet

5. Click:
   Ethernet

6. Click the active Ethernet connection.

7. Find:
   IP assignment

8. Click:
   Edit

9. Choose:
   Manual

10. Turn on:
    IPv4

11. Fill these values:

    IP address:
    192.168.10.1

    Subnet mask:
    255.255.255.0

    Gateway:
    leave empty

    DNS:
    leave empty

12. Click:
    Save

13. Use the USB connection one time to set the Thor Ethernet IP.

14. SSH over USB:
    ssh mta188@192.168.55.1

15. Set the Thor Ethernet IP:
    sudo ip addr add 192.168.10.2/24 dev enP2p1s0

16. Now SSH over Ethernet:
    ssh mta188@192.168.10.2


How to clone the code and install requirements
----------------------------------------------
1. SSH into Thor (open a Ubuntu terminal):
   ssh mta188@192.168.10.2

2. Clone the code:
   git clone https://github.com/ariya954/SFU_PhD.git
   cd “Projects/Gaussian Splatting“

3. If needed, initialize submodules:
   git submodule update --init --recursive

4. Load conda:
   source ~/miniconda3/etc/profile.d/conda.sh

5. Create the GS environment:
   conda create -n gs python=3.10 -y

6. Activate the GS environment:
   conda activate gs

7. Install requirements:
    pip install -r requirements.txt

8. After the code is cloned and the requirements are installed, use the next parts of this guide to compile, train, render, and check metrics.


How to compile the GS code
--------------------------
1. SSH into Thor (open a Ubuntu terminal):
   ssh mta188@192.168.10.2

2. Load conda:
   source ~/miniconda3/etc/profile.d/conda.sh

3. Activate the GS environment:
   conda activate gs

4. For the original code:
   cd ~/gaussian-splatting

5. For the reduction-tree code:
   cd ~/gaussian-splatting-reduction-tree

6. Delete old build files for diff-gaussian-rasterization:
   rm -rf submodules/diff-gaussian-rasterization/build
   rm -rf submodules/diff-gaussian-rasterization/*.egg-info
   rm -rf submodules/diff-gaussian-rasterization/*.so

7. Delete old build files for simple-knn:
   rm -rf submodules/simple-knn/build
   rm -rf submodules/simple-knn/*.egg-info
   rm -rf submodules/simple-knn/*.so

8. Compile diff-gaussian-rasterization:
   cd submodules/diff-gaussian-rasterization
   python setup.py build_ext --inplace
   cd ../..

9. Compile simple-knn:
   cd submodules/simple-knn
   python setup.py build_ext --inplace
   cd ../..


How to train
------------
1. SSH into Thor (open a Ubuntu terminal):
   ssh mta188@192.168.10.2

2. Load conda:
   source ~/miniconda3/etc/profile.d/conda.sh

3. Activate the GS environment:
   conda activate gs

4. For the original code:
   cd ~/gaussian-splatting

5. For the reduction-tree code:
   cd ~/gaussian-splatting-reduction-tree

6. Train the original code:
   python train.py -s /home/mta188/gaussian-splatting/data/tandt/truck

7. Train the reduction-tree code:
   python train.py -s /home/mta188/gaussian-splatting-reduction-tree/data/tandt/truck


How to render
-------------
1. SSH into Thor (open a Ubuntu terminal):
   ssh mta188@192.168.10.2

2. Load conda:
   source ~/miniconda3/etc/profile.d/conda.sh

3. Activate the GS environment:
   conda activate gs

4. For the original code:
   cd ~/gaussian-splatting

5. For the reduction-tree code:
   cd ~/gaussian-splatting-reduction-tree

6. Render:
   python render.py -m output/truck


How to check metrics
--------------------
1. SSH into Thor (open a Ubuntu terminal):
   ssh mta188@192.168.10.2

2. Load conda:
   source ~/miniconda3/etc/profile.d/conda.sh

3. Activate the GS environment:
   conda activate gs

4. For the original code:
   cd ~/gaussian-splatting

5. For the reduction-tree code:
   cd ~/gaussian-splatting-reduction-tree

6. Run metrics:
   python metrics.py -m output/truck


How to copy data/code
---------------------
Copy from laptop to Thor:
scp -r /path/to/local/folder mta188@192.168.10.2:/home/mta188/

Copy the reduction-tree repo from Thor to laptop:
scp -r mta188@192.168.10.2:/home/mta188/gaussian-splatting-reduction-tree ./

Copy the original repo from Thor to laptop:
scp -r mta188@192.168.10.2:/home/mta188/gaussian-splatting ./

Copy a single file from Thor to laptop:
scp mta188@192.168.10.2:/home/mta188/file.txt ./


How to create requirements files
--------------------------------
1. SSH into Thor (open a Ubuntu terminal):
   ssh mta188@192.168.10.2

2. Load conda:
   source ~/miniconda3/etc/profile.d/conda.sh

3. Activate the GS environment:
   conda activate gs

4. For the original code:
   cd ~/gaussian-splatting

5. For the reduction-tree code:
   cd ~/gaussian-splatting-reduction-tree

6. Create requirements.txt:
   python -m pip freeze > requirements.txt

7. Create conda environment file:
   conda env export --no-builds > environment.yml
