import scrapy


class RestaurantsSpider(scrapy.Spider):
    name = 'restaurants_trip_advisor'
    start_urls = ['https://www.tripadvisor.ru/Restaurants-g298484-Moscow_Central_Russia.html#MAINWRAP']
    custom_settings = {
        'FEED_URI': 'reviews.txt',
        'FEED_FORMAT': 'txt',
        'FEED_EXPORTERS': {
            'txt': 'txt_exporter.TxtItemExporter',
        }
    }

    def parse(self, response):
        for href in response.css('.listing h3 a::attr(href)'):
            full_url = response.urljoin(href.extract())
            yield scrapy.Request(full_url, callback=self.parse_restaurant)

    def parse_restaurant(self, response):
        for href in response.css('.reviewSelector .quote a::attr(href)'):
            review_url = response.urljoin(href.extract())
            yield scrapy.Request(review_url, callback=self.parse_review)

    def parse_review(self, response):
        yield {
            'review': " ".join(response.css('.first .entry p::text').extract()),
        }


