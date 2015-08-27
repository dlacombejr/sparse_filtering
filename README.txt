####################################################
################  EC2  SETUP  ######################
####################################################

Step 1) 

sudo apt-get --assume-yes install git
git init
git pull https://github.com/dlacombe2013/convDSF

Step 2)

chmod +x setup1.sh
Yes | ./setup1.sh

## wait a few seconds for reboot

Step 3) 

chmod +x setup2.sh
./setup2.sh

