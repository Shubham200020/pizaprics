df['extra_cheese']=df['extra_cheese'].replace(to_replace='no',value=0)
df['extra_cheese']=df['extra_cheese'].replace(to_replace='yes',value=1)
