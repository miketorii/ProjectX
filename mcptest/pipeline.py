import asyncio

class SkuNormalizer:
    async def transform(self, data):
        data['sku'] = data.get('sku','').strip().upper()
        print(data['sku'])        
        return data

class CurrencyConverter:
    def __init__(self, target="JPY"):
        self.target = target

    async def transform(self, data):
        data['price'] = f"{data.get('price',0)}{self.target}"
        print(data['price'])        
        return data

class StockStatusUnifier:
    async def transform(self, data):
        status = data.get('stock')
        data['is_available'] = True if status == "in_stock" else False
        print(data['is_available'])
        return data    

class PipeLine:
    def __init__(self):
        self.transformers = []

    def add(self, transformer):
        self.transformers.append(transformer)
        return self

    async def process(self, data):
        result = data
        for transformer in self.transformers:
            result = await transformer.transform(result)
        return result
        
async def func_main():
    print("-------func main start------")
    pipeline = PipeLine()

    raw_product_data = {
        "sku":"iPhone-15-Pro",
        "price":150000,
        "stock":"in_stock"
    }
    
    pipeline.add(SkuNormalizer()).add(CurrencyConverter(target="JPY")).add(StockStatusUnifier())

    unified_product = await pipeline.process(raw_product_data)

    print(unified_product)
    print("-------func main end------")    
                                      
if __name__ == "__main__":
    print("-------Start-------")
    asyncio.run(func_main())
    print("-------End-------")    
    
