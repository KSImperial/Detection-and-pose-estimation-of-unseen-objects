import os
path = "Path to MASKRCNN output masks"
fileCollection = next(os.walk(path))[1]
fileCollection =  ["\\" + suit for suit in fileCollection]
dirs = [m+str(n) for m,n in zip([path]*len(path), fileCollection)]


def renumber(oldElem,makeGTruth=False):
    members = [(1,2,3,4,11,12,13,14,15,16,24),(5,6,7,8,9),(10,25,26,27,28,29),(17,18,30),(19,20,21,22,23)]
    oldElem["obj_id"] = members.index([item for item in members if oldElem["obj_id"] in item][0])+1
    if makeGTruth:
        oldElem["pred_idx"] = oldElem["obj_id"]
    return oldElem
    

filename = "mask_rcnn_predict.yml"
files= [suit + "\\" + filename for suit in dirs]
#print(files)

import yaml
import json
import io
from collections import defaultdict
files = files[:] #define any subsections
filecount = 1
newStream = {}
dictList = []
for singleFile in files:
    with open(singleFile, "r") as stream:
        try:
            streamval = yaml.safe_load(stream)
            
            img_ids = list(streamval.keys())
            for imID in img_ids:
                newList = []
                obj_ids = list(streamval[imID].keys())
                for obID in obj_ids:
                    for nelem in streamval[imID][obID]:
                     nelem = renumber(nelem,True)
                     newList.append(nelem)
                     #print(nelem)

                for i in list(range(1,6)):
                    appendList = []
                    for e in newList:
                        
                        if int(e["obj_id"]) == i:
                            print(e)
                            if imID not in newStream:
                                newStream[imID] = {}
                            appendList.append(e)
                    if appendList != []:
                        newStream[imID][i] = appendList
                        thisdict = {"im_id": imID,"inst_count": len(appendList),"obj_id": i,"scene_id":filecount}
                        dictList.append(thisdict)
                        
                        

            json_object = json.dumps(dictList)
            with open(r"E:\OneDrive\OneDrive - King's College London\MSc Project\6dPose\General_Objects\re_annotated\data.json", "w") as outfile:
                json.dump(dictList, outfile)


            print("#"*200)
            #print(streamval)
        except yaml.YAMLError as exc:
            print(exc)
        filecount+=1
