# llama3.2-vision-ha-integrations
### ways to integrate Llama3.2 vision with home assistant. 
### i first want to have vision check an object detection from mqtt and 
### add metadata to log to the event like license plates, recognized people, 
### detailed descriptions of people, etc...

## USPS
### the usps-check script made me realize we can just ask llama3.2-vision if something is in the image. 
### we can pass a URL for a camera snapshot and the class of the object to find.
### then we are setup to create a home assistant sensor for the script and we 
### can do all the cool things
