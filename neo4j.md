连接服务器
ssh wangyuxiao@202.120.40.86 -p 1251
docker ps -a
找到neo4j
docker start 2549140c3403
映射到本地
ssh -L 7474:127.0.0.1:7474 wangyuxiao@202.120.40.86 -p 1251  
查看nuo4j日志
docker logs -f neo4j