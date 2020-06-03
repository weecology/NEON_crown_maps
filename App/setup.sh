#Docker Install and setup for NEON visualization
ssh <>
sudo docker pull visus/dataportal

#Open port and run
sudo docker run -it --rm -p 8080:80 visus/dataportal

#persist data
docker run -it --rm -p 8080:80 -v /my/data/folder:/data visus/dataportal

#change config, see config.js
#change internal server settings
sudo docker run -it --rm -p 8080:80 -v /home/bweinstein/config.js:/home/OpenVisus/dataportal/viewer/config.js visus/dataportal
