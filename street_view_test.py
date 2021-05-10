import requests
from street_view import StreetViewer
import matplotlib.pyplot as plt

# test with GWU marvin center address
gwu_viewer = StreetViewer(api_key=api_key,
                           location='800 N Main St., Tarboro NC 27886')
gwu_viewer.get_meta()
test_pic = gwu_viewer.get_pic()

plt.imshow(test_pic)
plt.show()

meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
pic_base = 'https://maps.googleapis.com/maps/api/streetview?'
api_key = 'AIzaSyBqm_ZCjRZpz2x8ScGazSDwVC4mdN6iSnQ'

# location = '800 N Main St., Tarboro NC 27886'

# # define the params for the metadata reques
# meta_params = {'key': api_key,
#                'location': location}
# # define the params for the picture request
# pic_params = {'key': api_key,
#               'location': location,
#               'size': "640x640"}

# # obtain the metadata of the request (this is free)
# meta_response = requests.get(meta_base, params=meta_params)

# print(meta_response.json())

# pic_response = requests.get(pic_base, params=pic_params)

# for key, value in pic_response.headers.items():
#     print(f"{key}: {value}")

# with open('test.jpg', 'wb') as file:
#     file.write(pic_response.content)
# # remember to close the response connection to the API
# pic_response.close()

# # using matpltolib to display the image
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# plt.figure(figsize=(10, 10))
# img=mpimg.imread('test.jpg')
# imgplot = plt.imshow(img)
# plt.show()
