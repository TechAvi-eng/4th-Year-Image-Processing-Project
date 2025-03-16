for i = 1:300

    d = extractdata(dlFeatures(:,:,i));
    if max(d, [], "all")>0.1
    
    imagesc(d); colorbar
    title(num2str(i))
    %set(gca,'ColorScale','log')

        pause(0.2)
    end

end


%%
for i = 1:13

    d = extractdata(ans(:,:,i));
    
    imagesc(d); colorbar
    title(num2str(i))
        caxis([0 1])

        pause(0.6)
end

%%
for i = 1:13

    d = (ans(:,:,i));
    
    imagesc(d); colorbar
    title(num2str(i))
            caxis([0 1])

        pause(0.5)

end