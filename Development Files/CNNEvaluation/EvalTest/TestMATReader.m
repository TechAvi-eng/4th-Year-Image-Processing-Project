function  out = TestMATReader(filename)

load(filename);


[im, masks] = resizeImageandMask(im, masks, [528, 704]);

out{1} = masks;
out{2} = label;
end