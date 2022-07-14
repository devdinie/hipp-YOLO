import os
import pandas as pd

def convert_annot_csv_to_yolo(annot_df,labeldict=dict(zip(["Cat_Face"], [0,])),
                              path="",target_name="data_train.txt", abs_path=False,):
        
        
    # Encode labels according to labeldict if code's don't exist
    if not "code" in annot_df.columns:
        annot_df["code"] = annot_df["label"].apply(lambda x: labeldict[x])
    # Round float to ints
    for col in annot_df[["xmin", "ymin", "xmax", "ymax"]]:
        annot_df[col] = (annot_df[col]).apply(lambda x: round(x))

    # Create Yolo Text file
    last_image = ""
    txt_file = ""
    
    for index, row in annot_df.iterrows():
        if not last_image == row["image"]:
            if abs_path:
                txt_file += "\n" + row["image_path"] + " "
            else:
                txt_file += "\n" + os.path.join(path, row["image"]) + " "
            txt_file += ",".join([str(x) for x in (row[["xmin", "ymin", "xmax", "ymax", "code"]].tolist())])
        else:
            txt_file += " "
            txt_file += ",".join([str(x) for x in (row[["xmin", "ymin", "xmax", "ymax", "code"]].tolist())])
        last_image = row["image"]
    
    file = open(target_name, "w")
    file.write(txt_file[1:])
    file.close()
    return True