#SJER 
d<-read.csv("/Users/Ben/Downloads/SJER_2019.csv")
dim(d)


m<-d  %>% mutate(log_area=log(area), log_height=log(height)) %>% lm(data=.,log_area~log_height)
summary(m)
field_predict<-data.frame(x=seq(3,50,1),y=exp(predict(m,newdata = data.frame(log_height=log(seq(3,50,1))))))

ggplot(d,aes(x=height,area)) + geom_hex() + geom_line(data=field_predict,aes(x=x,y=y)) + stat_smooth(method = "lm")

d %>% filter(score>0.5) %>% ggplot(.,aes(x=log(height),log(area))) + geom_hex() + stat_smooth(method = "lm")

d %>% filter(geo_index =="260000_4112000",height < 10, score>0.7) %>% ggplot(.,aes(x=height,area)) + geom_point() + stat_smooth(method = "lm")


o<-read.csv("/Users/Ben/Downloads/OSBS_2019.csv")
dim(o)
o %>% filter(height<40) %>% ggplot(.,aes(x=height,area)) + geom_hex() + stat_smooth(method = "lm")
o %>% filter(height<40, score>0.7) %>% ggplot(.,aes(x=height,area)) + geom_hex(aes(fill=..count..)) + stat_smooth(method = "lm") + scale_x_continuous(n.breaks = 10,limits=c(0,40))

