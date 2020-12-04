#Docker Install and setup for NEON visualization
ssh <>
sudo docker pull visus/dataportal


#Azcopy login
sudo azcopy login --tenant-id 6689f796-92e8-413e-864a-db37d7fca00b

azcopy copy OpenVisus/ https://neon.blob.core.windows.net/openvisus --recursive

#if this fails create a shared access token under the blob container 'sas' token must be a quotes.

#change config, see config.js
#change internal server settings
sudo docker run --rm -d -p 8080:80 -v /datadrive/OpenVisus/:/converted -v /home/bweinstein/NEON_crown_maps/App/config.js:/home/OpenVisus/dataportal/ext/visus/config.js visus/dataportal
