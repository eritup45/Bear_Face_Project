import cv2
import easygui
import os
def set_cam():
    filepath = "setting.txt"
    if os.path.isfile(filepath):
        print("setting.txt存在。")
        f = open('setting.txt','r')
        #print(type(f.read()))
        return int(f.read()) 
    else:
        print("setting.txt不存在。")
    for i in range(10):
        camera_check= cv2.VideoCapture(i).isOpened()#Returns true if video capturing has been initialized already.
        next_camera_check= cv2.VideoCapture(i+1).isOpened()
        print('i :',i, 'check: ',camera_check)
        print('i+1 :',i+1, 'check: ',next_camera_check)
        cap = cv2.VideoCapture(i)
        #print("check:",i, camera_check)
        while(True):
            ret, frame = cap.read()
            if camera_check == False:
                print("error")
            cv2.imshow('frame', frame)
            value = easygui.ynbox("Is this the video camera?")
            cap.release()
            cv2.destroyAllWindows() 
            break
                #print("ans : i == ",i)
        if value == True:
            fp = open("setting.txt", "a")
            fp.write(str(i))
            fp.close()
            #print("insert")
            return i 
        else:
            continue
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #   break
    #print("camera",camera_check)
   

if __name__ == "__main__":
    set_cam()