from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="Vier_Onderdelen_Wit")
trainer.setTrainConfig(object_names_array=["Schoepen_groot","Schijf_zilver","Kom","Rotor"], batch_size=4, num_experiments=100, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()

trainer.setDataDirectory(data_directory="Vier_Onderdelen_Wit")
evaluate = trainer.evaluateModel(model_path="Vier_Onderdelen_Wit/models", json_path="Vier_Onderdelen_Wit/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)

x = []
for i in range(len(evaluate)):
	map = evaluate[i]['map']
	experiment = evaluate[i]['model_file']
	x.append((map, experiment))
	x.sort()
  
text_file = open("maps_vier_onderdelen_wit.txt", "wt")
for i in x:
  e = text_file.writelines(str(i)+  "\n")
text_file.close()