#  EC2  SETUP 

1. Install Git and pull SF repository

    sudo apt-get --assume-yes install git
    git init
    git pull https://github.com/dlacombe2013/convDSF

2. Install first set of dependencies

    chmod +x setup1.sh
    Yes | ./setup1.sh

    wait a few seconds for reboot...

3. Install second set of dependencies

    chmod +x setup2.sh
    ./setup2.sh

