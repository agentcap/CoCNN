for i=1:685
    
    load('label_map.mat');
    arr = bac_arr;
   
    label_dir = 'all_images_output';
    img = imdecode(predictions_paperdoll(i).labeling,'png');
    img_name = [label_dir '/' num2str(i) '.png'];
    
    for j=1:56
        img(img==j) = arr(j);
    end
    
    imwrite(img, img_name)
    
end