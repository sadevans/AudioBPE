# AudioBPE

```
pip install -r requirements.txt
```


- `download_dataset.py` - скачивание датасета и сохранение flac файлов.
- `get_embeddings.py` - олучение эмбедингов HuBERT, сохранение эсбедингов. 
- `process_dataset.py` - обработка flac файлов адаптивным токенизатором с целью разбиения на сегменты, сохранение сегментов, получение эмбедингов HuBERT, сохранение эсбедингов.
