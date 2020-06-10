#Docker Install and setup for NEON visualization
ssh <>
sudo docker pull visus/dataportal

#change config, see config.js
#change internal server settings
sudo docker run -it --rm -p 8080:80 -v /home/bweinstein/config.js:/home/OpenVisus/dataportal/ext/visus/config.js visus/dataportal
