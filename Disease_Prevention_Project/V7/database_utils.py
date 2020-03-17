import numpy as np
import json

# Convert np to json, in order to store in sqlite
def adapt_list(list_of_face_encodings):
    cvt_list_of_face_encodings = []
    for i in range(len(list_of_face_encodings)):
        cvt_list_of_face_encodings.append(
            json.dumps(list_of_face_encodings[i].tolist()))
        # cvt_list_of_face_encodings = json.dumps(list_of_face_encodings[i].tolist())

    # TODO:檢查db裡None是否被存成字串'None'
    if len(list_of_face_encodings) == 0:
        cvt_list_of_face_encodings = None
    return cvt_list_of_face_encodings

# Convert json to list
def convert_list(cvt_list_of_face_encodings):
    # my_list = []

    # Can't find face
    if cvt_list_of_face_encodings == 'None':
        return None
    elif cvt_list_of_face_encodings == None:
        return None
    else:
        my_list = np.array(json.loads(cvt_list_of_face_encodings))
    return my_list
