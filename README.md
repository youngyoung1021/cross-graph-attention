#Scene Graph Extraction  
## DramaQA 지식체계 추출


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

"idx_to_files": 각 이미지의 경로를 뜻합니다. 
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

        [31,32]: [custom_data_info["ind_to_classes"][32],custom_data_info["ind_to_classes"][32]]
        --> ['engine','bus'] --> engine과 bus가 연결되어있음
    
    "rel_labels(sorted)": 연결된 노드간의 엣지 label입니다.(최대 6320개)
    ex) [20, 20, 20, 20, 40, 40, 40 ... ]
        [20] --> [custom_data_info["ind_to_predicates"][20] --> "has"

    "rel_scores(sorted)": 엣지의 prediction score입니다.(최대 6320개)
     ex) [0.9488074779510498, 0.946490466594696, ... , 0.9349974393844604]
    }
```
