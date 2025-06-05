function  out = SegMATReader(filename)

load(filename);


%[~, masks] = resizeImageandMask(ones([520, 704]), masks, [528, 704]);
%bbox(:,2) = min(bbox(:,2)+4, 528); %will imlpement into resize function later, see bug report


out{1} = masks;
out{2} = boxLabel;
out{3} = boxScore;
end