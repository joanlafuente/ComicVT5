
#!/bin/bash

mkdir ./data
mkdir ./logs

# download page images
wget https://obj.umiacs.umd.edu/comics/raw_page_images.tar.gz

# download ocr 
wget https://obj.umiacs.umd.edu/comics/COMICS_ocr_file.csv -P ./data/

# download ad pages to filter out
wget https://obj.umiacs.umd.edu/comics/predadpages.txt -P ./data/

# download panel images
wget https://obj.umiacs.umd.edu/comics/raw_panel_images.tar.gz -P ./data/

# untar 
tar -xvzf ./data/raw_panel_images.tar.gz -C ./data/
