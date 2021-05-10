import requests
import json
from io import BytesIO
from PIL import Image


class StreetViewer(object):
    def __init__(self, api_key, location, size="256x256",
                 folder_directory='./', verbose=True):
        """
        This class handles a single API request to the Google Static Street View API
        api_key: obtain it from your Google Cloud Platform console
        location: the address string or a (lat, lng) tuple
        size: returned picture size. maximum is 640*640
        folder_directory: directory to save the returned objects from request
        verbose: whether to print the processing status of the request
        """
        # input params are saved as attributes for later reference
        self._key = api_key
        self.location = location
        self.size = size
        self.folder_directory = folder_directory
        # call parames are saved as internal params
        self._meta_params = dict(key=self._key,
                                location=self.location)
        self._pic_params = dict(key=self._key,
                               location=self.location,
                               size=self.size)
        self.verbose = verbose

    def get_meta(self):
        """
        Method to query the metadata of the address
        """
        self._meta_response = requests.get(
            'https://maps.googleapis.com/maps/api/streetview/metadata?',
            params=self._meta_params)
        # turning the contents as meta_info attribute
        self.meta_info = self._meta_response.json()
        # meta_status attribute is used in get_pic method to avoid
        # query when no picture will be available
        self.meta_status = self.meta_info['status']
        if self._meta_response.ok:
            pass
            # if self.verbose:
            #     print(">>> Obtained Meta from StreetView API:")
            #     print(self.meta_info)
            # with open(self.meta_path, 'w') as file:
            #     json.dump(self.meta_info, file)
        else:
            print("Could not obtain the metadata, so address not readable")
        self._meta_response.close()

    def get_pic(self):
        """
        Method to query the StreetView picture and save to local directory
        """
        # only when meta_status is OK will the code run to query picture (cost incurred)
        if self.meta_status == 'OK':
            if self.verbose:
                print(">>> Picture available, requesting now...")
            self._pic_response = requests.get(
                'https://maps.googleapis.com/maps/api/streetview?',
                params=self._pic_params)
            self.pic_header = dict(self._pic_response.headers)
            if self._pic_response.ok:
                if self.verbose:
                    print(f">>> Loading objects as PIL.Image")
                byte_object = BytesIO(self._pic_response.content)
                im = Image.open(byte_object)
                self._pic_response.close()
                return im
            else:
                return 0
        else:
            return 0
