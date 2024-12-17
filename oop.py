
class person:
    national="Viet Nam"
  
    def __init__(self,name,age):
            self.name=name
            self.age=age
    def infor(self):
            print(self.name+" "+self.age)
if __name__=="__main__":
    p=person("anh ba",10)
    print(p.national,p.name ,p.age)