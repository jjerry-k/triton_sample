# Triton Inference Server Example

## Quick Setting
> `Quick Setting` run on cpu.  
> `Model format` is `TorchScript`.

1. Download Model 
- Download Link: https://drive.google.com/drive/folders/1QeKk-7v-JzyG0v8wquybT7gjJatsg4EJ?usp=sharing

2. Move `model.pt` file to `model_repository/{model name}/1`

3. Run docker compose
```bash
docker compose up
```

## To Do List
- [ ] GPU Mode
- [ ] Detection router
  - [ ] Postprocessing
- [ ] Segmentation router
- [ ] Variable input type
- [ ] Variable output type
