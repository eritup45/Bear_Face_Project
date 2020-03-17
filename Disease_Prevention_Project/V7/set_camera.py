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
        camera_check= cv2.VideoCapture(i).isOpened()
        if camera_check == False:
            print("error")
            break
        cap = cv2.VideoCapture(i)
        #print("check:",i, camera_check)
        while(True):
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            value = easygui.ynbox("Is this the video camera?")
                #print("ans : i == ",i)
            break
        if value == True:
            fp = open("setting.txt", "a")
            fp.write(str(i))
            fp.close()
            cap.release()
            cv2.destroyAllWindows()
            #print("insert")
            return i 
        else:
            continue
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #   break


        
    #print("camera",camera_check)
   

if __name__ == "__main__":
    set_cam()