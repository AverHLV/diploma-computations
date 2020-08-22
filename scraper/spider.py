from scrapy.crawler import CrawlerProcess
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

from hashlib import md5
from settings import spider_settings


def run_spider():
    """ Run VoteSpider as new process """

    process = CrawlerProcess(spider_settings)
    process.crawl(VoteSpider)
    process.start()


class VoteSpider(CrawlSpider):
    """ Rada.gov.ua crawl spider for scraping all vote results from all sessions of the VIII convocation """

    name = 'votes'
    allowed_domains = 'rada.gov.ua',

    start_urls = 'http://w1.c1.rada.gov.ua/pls/zweb2/webproc2_5_1_J?' \
                 'ses=10009&num_s=2&num=&date1=&date2=&name_zp=&out_type=&id=',

    rules = (
        Rule(LinkExtractor(
            allow=[
                r'webproc2_5_1_J\?ses=10009&num_s=2&num=&date1=&date2=&name_zp=&out_type=&id=&page=\d{1,3}&zp_cnt=20',
                r'webproc4_1\?pf3511=\d{5}',
                r'/pls/radan_gs09/ns_zakon_gol_dep_wohf\?zn=.{1,}',
                r'/pls/radan_gs09/ns_golos\?g_id=.{1,}'
            ]),

            follow=True),

        Rule(LinkExtractor(
            allow=r'http://w1\.c1\.rada\.gov\.ua/pls/radan_gs09/ns_golos_rtf\?g_id=.{1,}&vid=0'),
            callback='save_vote_results'
        )
    )

    def __init__(self, save_path='data/{0}.rtf', *args, **kwargs):
        """
        VoteSpider initialization

        :param save_path: relative path for saving scraped files
        """

        super().__init__(*args, **kwargs)
        self.file_number = -1
        self.save_path = save_path
        self.hashes = set()

    def save_vote_results(self, response):
        """ Parse response and save vote results to file """

        body_hash = md5()
        body_hash.update(response.body)
        body_hash = body_hash.hexdigest()

        hashes_length = len(self.hashes)
        self.hashes.add(body_hash)

        # check duplicates by md5 hash

        if hashes_length == len(self.hashes):
            self.logger.info('Duplicate file with hash: {0}'.format(body_hash))
            return

        self.file_number += 1
        filename = self.save_path.format(self.file_number)

        with open(filename, 'wb') as file:
            file.write(response.body)

        self.logger.info('File saved: {0}, from: {1}'.format(filename, response.url))


if __name__ == '__main__':
    run_spider()
    print('Done')
