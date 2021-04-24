# coding: utf-8
class Car
  def initialize(carname, modelname)
    @name = carname
    @model = modelname
    @engine = 2000
  end

  def display
    puts(@name)
    puts(@model)
    puts(@engine)
  end

  attr_accessor :engine
  attr_reader:name
  attr_reader:model  
  
end

class Soarer < Car
  def openroof
    puts("opened")
  end
end

car1 = Car.new("crown", "toyota")
car1.display

car1.engine = 1500
car1.display

puts(car1.model)

car2 = Soarer.new("soarer", "toyota")
car2.openroof
car2.display





