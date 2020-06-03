#Docker Install and setup for NEON visualization
ssh <>
sudo docker pull visus/dataportal

#Open port?
sudo docker run -it --rm -p 8080:80 visus/dataportal
