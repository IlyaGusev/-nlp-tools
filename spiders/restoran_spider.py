import scrapy
from scrapy.contrib.exporter import BaseItemExporter


class TxtItemExporter(BaseItemExporter):
    def __init__(self, file, **kwargs):
        self._configure(kwargs, dont_fail=True)
        self.file = file

    def export_item(self, item):
        output = '%s\n' % item['review']
        self.file.write(str.encode(output))


class RestaurantsSpider(scrapy.Spider):
    name = 'restaurants_restoran'
    base = 'https://www.restoran.ru'
    catalog_page = 1
    catalog_max_pages = 60
    reviews_page = 1
    start_urls = ['https://www.restoran.ru/msk/catalog/restaurants/all/']
    custom_settings = {
        'FEED_URI': 'reviews_restoran.txt',
        'FEED_FORMAT': 'txt',
        'FEED_EXPORTERS': {
            'txt': 'restoran_spider.TxtItemExporter',
        }
    }

    def parse(self, response):
        for href in response.css('.item h2 a::attr(href)'):
            reviews_url = response.urljoin(href.extract()).replace("detailed", "opinions")
            yield scrapy.Request(reviews_url, callback=self.parse_reviews)
        self.catalog_page += 1
        if self.catalog_page <= self.catalog_max_pages:
            next_page_url = self.start_urls[0] + "?page=" + str(self.catalog_page)
            yield scrapy.Request(next_page_url, callback=self.parse)

    def parse_reviews(self, response):
        for text in response.xpath("//div[@class='reviews-list']/div[@class='item']/div[contains(@style,'display:none;')]/p/text()").extract():
            yield {
                'review': text.replace("\n", " ")
            }
        refs = response.css('div.navigation a::attr(href)').extract()
        if refs:
            href = refs[-1]
            if href.find('java') == -1:
                next_page_url = self.base + href
                yield scrapy.Request(next_page_url, callback=self.parse_reviews)




