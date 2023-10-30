# DramaQA 지식체계 추출
   
### 도커 이미지 및 가상환경
   
```bash
docker pull oodsdsd/scene_graph_extract
docker run -it --gpus "device=0" --name scene_graph_extract oodsdsd/scene_graph_extract /bin/bash
conda activate scene_graph_extract
cd /home/scene_graph_extraction
```  
        
### 이미지 경로 설정

```bash
지식체계를 추출하고자하는 이미지 디렉토리를 /scene_graph_extraction/{directory}에 추가해주세요.

ex) /scene_graph_extraction/dramaqa_frames/AnotherMissOh15/003/0080
``` 
        
### 지식체계 추출

```bash
### 사용자 지정 Frame 추출
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 ./tools/relation_test_net.py  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./checkpoints/upload_casual_motif_sgdet/glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/upload_casual_motif_sgdet OUTPUT_DIR ./checkpoints/upload_casual_motif_sgdet TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH /your_custom_image_path DETECTED_SGG_DIR /your_output_path

### DramaQA 대한 전체 Frame 추출 -> input image path에 output 저장
sh ./scene_graph_extract.sh
```

### Input / Output path
output으로 custom_data_info.json, custom_prediction.json 생성됩니다.
```bash
ex)
input image path: 
|-- AnotherMissOh15
|   `-- 003
|       `-- 0080
|           |-- IMAGE_0000008566.jpg
|           |-- IMAGE_0000008574.jpg
|           |-- IMAGE_0000008582.jpg
|           ...
|           |-- IMAGE_0000008694.jpg
|           |-- IMAGE_0000008702.jpg

output path:
|-- AnotherMissOh15
|   `-- 003
|       `-- 0080
|           |-- IMAGE_0000008566.jpg
|           |-- IMAGE_0000008574.jpg
|           |-- IMAGE_0000008582.jpg
|           ...
|           |-- IMAGE_0000008694.jpg
|           |-- IMAGE_0000008702.jpg
|           |-- custom_data_info.json
|           |-- custom_prediction.json

```

### Output detail
**custom_data_info.json**
```bash
custom_data_info.keys():

"idx_to_files": 폴더내에 각 이미지의 경로를 뜻합니다. 
ex)["./dramaqa_frames/AnotherMissOh15/003/0080/IMAGE_0000008630.jpg",
    "./dramaqa_frames/AnotherMissOh15/003/0080/IMAGE_0000008574.jpg"
    ...
    "./dramaqa_frames/AnotherMissOh15/003/0080/IMAGE_0000008702.jpg"]

"ind_to_classes": Detect된 object 클래스 입니다. (150 classes)
ex) ["airplane", "animal", "arm", "bag", ... , "basket", "beach"]

"ind_to_predicates": Detecte된 object들 간의 edge 클래스 입니다.(51 classes)
ex) ["above", "across", "against", "along", ..., "walking on", "watching"]
```
**custom_prediction.json**
```bash
custom_prediction.keys():

모든 값들은 prediction score 값으로 정렬되어있습니다.
ex)
{"0": custom_data_info의 각 이미지별 경로입니다. 
      ex) custom_data_info["idx_to_files"][0]
      위의 custom_data_info 예시에서 "./dramaqa_frames/AnotherMissOh15/003/0080/IMAGE_0000008630.jpg" 입니다.

    "bbox"(sorted): detect된 object의 bbox(a,b,c,d 좌표)입니다.(최대 80개)
    ex)[[662.5, 126.625, 783.5, 452.25], [178.75, 272.0, 346.75, 599.0], ... , [1.4052734375, 349.25, 192.875, 436.25], [665.5, 127.5, 740.5, 190.0]]
    
    "bbox_labels"(sorted): detect된 object의 bbox label입니다.(최대 80개)
    ex) [149, 149, 149, 17, ..., 111, 145, 126, 11]
        
        149: custom_data_info["ind_to_classes"][149] --> "woman"
        17: custom_data_info["ind_to_classes"][17] --> "bottle"
        ...
        111: custom_data_info["ind_to_classes"][111] --> "shirt"

    "bbox_scores(sorted)": detect된 object의 bbox별 prediction score입니다.(최대 80개)
    ex) [0.4657273292541504, 0.4292507469654083, ... , 0.284429669380188, 0.2633313536643982]


    ### 각 엣지는 모든 노드와 연결되어있는 것을 가정하기 때문에 (전체 노드수 -1 * 전체 노드 수) 만큼 나올 수 있습니다. 
    ### 하지만 이는 prediction relation score가 낮은 엣지도 많기 때문에 score가 0.5 이상인것만 DramaQA 데이터셋으로 전달드립니다. 
    ### rel_score > 0.5
    "rel_pairs(sorted)": detect된 bbox(노드)간의 연결되어있는 pair를 나타냅니다.(최대 6320개)
    ex)[[31, 32], [33, 32], [9, 32], [23, 32], ... , [42, 23], [61, 23], [47, 25], [25, 51]]

        [31,32]: [custom_data_info["ind_to_classes"][31],custom_data_info["ind_to_classes"][32]]
        --> ['engine','bus'] --> engine과 bus가 연결되어있음
    
    "rel_labels(sorted)": 연결된 노드간의 엣지 label입니다.(최대 6320개)
    ex) [20, 20, 20, 20, 40, 40, 40 ... ]
        [20] --> [custom_data_info["ind_to_predicates"][20] --> "has"

    "rel_scores(sorted)": 엣지의 prediction score입니다.(최대 6320개)
     ex) [0.9488074779510498, 0.946490466594696, ... , 0.9349974393844604]
    }
```

### 시각화

**Input Image**

![image](https://user-images.githubusercontent.com/44778298/202831042-cffb4f98-7409-452c-ae4b-521df7b78b22.png)

**Output Image**

![image](https://user-images.githubusercontent.com/44778298/202831059-51d73a05-022b-445f-994c-caf4d8f6eca6.png)


```bash
**************************************************
box_labels 0: shirt; score: 0.632802426815033
box_labels 1: woman; score: 0.47097983956336975
box_labels 2: man; score: 0.39938557147979736
box_labels 3: shirt; score: 0.3945291042327881
box_labels 4: hair; score: 0.3771052956581116
box_labels 5: pant; score: 0.3428615927696228
box_labels 6: table; score: 0.32002344727516174
box_labels 7: shelf; score: 0.30289050936698914
box_labels 8: bottle; score: 0.299174040555954
box_labels 9: shelf; score: 0.2271135151386261
**************************************************
rel_labels 0: 2_man => at => 6_table; score: 0.8761001229286194
rel_labels 1: 2_man => at => 7_shelf; score: 0.7776363492012024
rel_labels 2: 1_woman => at => 6_table; score: 0.7722375392913818
rel_labels 3: 3_shirt => at => 6_table; score: 0.7715178728103638
rel_labels 4: 1_woman => at => 7_shelf; score: 0.756068229675293
rel_labels 5: 2_man => watching => 1_woman; score: 0.7519111633300781
rel_labels 6: 2_man => wearing => 5_pant; score: 0.7431520819664001
rel_labels 7: 2_man => at => 9_shelf; score: 0.7313538789749146
rel_labels 8: 1_woman => wearing => 3_shirt; score: 0.7244991064071655
rel_labels 9: 1_woman => at => 9_shelf; score: 0.6979193687438965
rel_labels 10: 2_man => wearing => 0_shirt; score: 0.6863403916358948
rel_labels 11: 1_woman => wearing => 5_pant; score: 0.6432791948318481
rel_labels 12: 2_man => wearing => 3_shirt; score: 0.6212118268013
...
rel_labels 16: 5_pant => at => 6_table; score: 0.5563212037086487
rel_labels 17: 2_man => looking at => 8_bottle; score: 0.5294371247291565
rel_labels 18: 6_table => in front of => 2_man; score: 0.4990682005882263
rel_labels 19: 1_woman => with => 4_hair; score: 0.47303032875061035
```
