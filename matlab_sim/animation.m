function [] = animation(tt,xxc,zzc,angle_total,l_total,dxxc)
%ANIMATION do animation
%   input:tt --> total time span
%         xxc --> total x position of mass
%         zzc --> total z position of mass
%         angle --> total theta data
%         l_total --> total leg length data

figure(9);
for i=1:100:length(tt)
    cla;
    xc = xxc(i);
    zc = zzc(i);
    leg = l_total(i);
    angle = angle_total(i);
    
    draw_Leg(tt,xc,zc,leg,angle,i,dxxc);
    pause(0.00000001);   
    drawnow;
    
end
end

