from math import *
if __name__=='__main__':
    n,x=map(int,input().split())
    
    a=list(map(int,input().split()))
    l,r=0,n-1
    cnt=0
    a.sort()
    while(l<=r):
        if(a[l]+a[r]<=x):
            cnt+=1
            l=l+1
            r=r-1
        else:
            cnt=cnt+1
            r=r-1
    print(cnt)
        

   
   
   

   
    
    
    
    
   
    