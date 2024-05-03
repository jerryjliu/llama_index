from llama_index.core.prompts import MessageRole
from llama_index.llms.cohere import DocumentMessage
from llama_index.llms.cohere.utils import _message_to_cohere_documents


def test_message_to_cohere_documents():
    # Empty message
    message = DocumentMessage(role=MessageRole.USER, content="", additional_kwargs={})
    assert _message_to_cohere_documents(message) == []

    # Message with single document
    content = "==== Brooklyn ====\nBrooklyn (Kings County), on the western tip of Long Island, is the city's most populous borough. Brooklyn is known for its cultural, social, and ethnic diversity, an independent art scene, distinct neighborhoods, and a distinctive architectural heritage. Downtown Brooklyn is the largest central core neighborhood in the Outer Boroughs. The borough has a long beachfront shoreline including Coney Island, established in the 1870s as one of the earliest amusement grounds in the U.S. Marine Park and Prospect Park are the two largest parks in Brooklyn. Since 2010, Brooklyn has evolved into a thriving hub of entrepreneurship and high technology startup firms, and of postmodern art and design.\n\n\n==== Queens ====\nQueens (Queens County), on Long Island north and east of Brooklyn, is geographically the largest borough, the most ethnically diverse county in the United States, and the most ethnically diverse urban area in the world. Historically a collection of small towns and villages founded by the Dutch, the borough has since developed both commercial and residential prominence. Downtown Flushing has become one of the busiest central core neighborhoods in the outer boroughs. Queens is the site of the Citi Field baseball stadium, home of the New York Mets, and hosts the annual U.S. Open tennis tournament at Flushing Meadows–Corona Park. Additionally, two of the three busiest airports serving the New York metropolitan area, John F. Kennedy International Airport and LaGuardia Airport, are in Queens. The third is Newark Liberty International Airport in Newark, New Jersey.\n\n\n==== The Bronx ====\nThe Bronx (Bronx County) is both New York City's northernmost borough, and the only one that is mostly on the mainland. It is the location of Yankee Stadium, the baseball park of the New York Yankees, and home to the largest cooperatively-owned housing complex in the United States, Co-op City. It is also home to the Bronx Zoo, the world's largest metropolitan zoo, which spans 265 acres (1.07 km2) and houses more than 6,000 animals. The Bronx is also the birthplace of hip hop music and its associated culture. Pelham Bay Park is the largest park in New York City, at 2,772 acres (1,122 ha."
    message = DocumentMessage(
        role=MessageRole.USER,
        content="file_path: nyc_text.txt" + "\n\n" + content,
        additional_kwargs={},
    )
    res = _message_to_cohere_documents(message)
    assert (
        len(res) == 1
        and res[0]["source"] == "nyc_text.txt"
        and res[0]["text"] == content
    )

    # Message with multiple documents
    message = DocumentMessage(
        role=MessageRole.USER,
        content="file_path: nyc_text.txt\n\n==== Brooklyn ====\nBrooklyn (Kings County), on the western tip of Long Island, is the city's most populous borough. Brooklyn is known for its cultural, social, and ethnic diversity, an independent art scene, distinct neighborhoods, and a distinctive architectural heritage. Downtown Brooklyn is the largest central core neighborhood in the Outer Boroughs. The borough has a long beachfront shoreline including Coney Island, established in the 1870s as one of the earliest amusement grounds in the U.S. Marine Park and Prospect Park are the two largest parks in Brooklyn. Since 2010, Brooklyn has evolved into a thriving hub of entrepreneurship and high technology startup firms, and of postmodern art and design.\n\n\n==== Queens ====\nQueens (Queens County), on Long Island north and east of Brooklyn, is geographically the largest borough, the most ethnically diverse county in the United States, and the most ethnically diverse urban area in the world. Historically a collection of small towns and villages founded by the Dutch, the borough has since developed both commercial and residential prominence. Downtown Flushing has become one of the busiest central core neighborhoods in the outer boroughs. Queens is the site of the Citi Field baseball stadium, home of the New York Mets, and hosts the annual U.S. Open tennis tournament at Flushing Meadows–Corona Park. Additionally, two of the three busiest airports serving the New York metropolitan area, John F. Kennedy International Airport and LaGuardia Airport, are in Queens. The third is Newark Liberty International Airport in Newark, New Jersey.\n\n\n==== The Bronx ====\nThe Bronx (Bronx County) is both New York City's northernmost borough, and the only one that is mostly on the mainland. It is the location of Yankee Stadium, the baseball park of the New York Yankees, and home to the largest cooperatively-owned housing complex in the United States, Co-op City. It is also home to the Bronx Zoo, the world's largest metropolitan zoo, which spans 265 acres (1.07 km2) and houses more than 6,000 animals. The Bronx is also the birthplace of hip hop music and its associated culture. Pelham Bay Park is the largest park in New York City, at 2,772 acres (1,122 ha).\n\nfile_path: nyc_text.txt\n\nAnnually, the city averages 234 days with at least some sunshine. The city lies in the USDA 7b plant hardiness zone.Winters are chilly and damp, and prevailing wind patterns that blow sea breezes offshore temper the moderating effects of the Atlantic Ocean; yet the Atlantic and the partial shielding from colder air by the Appalachian Mountains keep the city warmer in the winter than inland North American cities at similar or lesser latitudes such as Pittsburgh, Cincinnati, and Indianapolis. The daily mean temperature in January, the area's coldest month, is 33.3 °F (0.7 °C). Temperatures usually drop to 10 °F (−12 °C) several times per winter, yet can also reach 60 °F (16 °C) for several days even in the coldest winter month. Spring and autumn are unpredictable and can range from cool to warm, although they are usually mild with low humidity. Summers are typically hot and humid, with a daily mean temperature of 77.5 °F (25.3 °C) in July.Nighttime temperatures are often enhanced due to the urban heat island effect. Daytime temperatures exceed 90 °F (32 °C) on average of 17 days each summer and in some years exceed 100 °F (38 °C), although this is a rare achievement, last occurring on July 18, 2012. Similarly, readings of 0 °F (−18 °C) are also extremely rare, last occurring on February 14, 2016. Extreme temperatures have ranged from −15 °F (−26 °C), recorded on February 9, 1934, up to 106 °F (41 °C) on July 9, 1936; the coldest recorded wind chill was −37 °F (−38 °C) on the same day as the all-time record low. The record cold daily maximum was 2 °F (−17 °C) on December 30, 1917, while, conversely, the record warm daily minimum was 87 °F (31 °C), on July 2, 1903.\n\nfile_path: nyc_text.txt\n\n==== The Bronx ====\nThe Bronx (Bronx County) is both New York City's northernmost borough, and the only one that is mostly on the mainland. It is the location of Yankee Stadium, the baseball park of the New York Yankees, and home to the largest cooperatively-owned housing complex in the United States, Co-op City. It is also home to the Bronx Zoo, the world's largest metropolitan zoo, which spans 265 acres (1.07 km2) and houses more than 6,000 animals. The Bronx is also the birthplace of hip hop music and its associated culture. Pelham Bay Park is the largest park in New York City, at 2,772 acres (1,122 ha).\n\n\n==== Staten Island ====\nStaten Island (Richmond County) is the most suburban in character of the five boroughs. Staten Island is connected to Brooklyn by the Verrazzano-Narrows Bridge, and to Manhattan by way of the free Staten Island Ferry, a daily commuter ferry that provides unobstructed views of the Statue of Liberty, Ellis Island, and Lower Manhattan. In central Staten Island, the Staten Island Greenbelt spans approximately 2,500 acres (10 km2), including 28 miles (45 km) of walking trails and one of the last undisturbed forests in the city. Designated in 1984 to protect the island's natural lands, the Greenbelt comprises seven city parks.\n\n\n\n\n\n=== Architecture ===\n\nNew York has architecturally noteworthy buildings in a wide range of styles and from distinct time periods, from the Dutch Colonial Pieter Claesen Wyckoff House in Brooklyn, the oldest section of which dates to 1656, to the modern One World Trade Center, the skyscraper at Ground Zero in Lower Manhattan and the most expensive office tower in the world by construction cost.Manhattan's skyline, with its many skyscrapers, is universally recognized, and the city has been home to several of the tallest buildings in the world. As of 2019, New York City had 6,455 high-rise buildings, the third most in the world after Hong Kong and Seoul. Of these, as of 2011, 550 completed structures were at least 330 feet (100 m) high, with more than fifty completed skyscrapers taller than 656 feet (200 m).\n\nfile_path: nyc_text.txt\n\nThe storm and its profound impacts have prompted the discussion of constructing seawalls and other coastal barriers around the shorelines of the city and the metropolitan area to minimize the risk of destructive consequences from another such event in the future.The coldest month on record is January 1857, with a mean temperature of 19.6 °F (−6.9 °C) whereas the warmest months on record are July 1825 and July 1999, both with a mean temperature of 81.4 °F (27.4 °C). The warmest years on record are 2012 and 2020, both with mean temperatures of 57.1 °F (13.9 °C). The coldest year is 1836, with a mean temperature of 47.3 °F (8.5 °C). The driest month on record is June 1949, with 0.02 inches (0.51 mm) of rainfall. The wettest month was August 2011, with 18.95 inches (481 mm) of rainfall. The driest year on record is 1965, with 26.09 inches (663 mm) of rainfall. The wettest year was 1983, with 80.56 inches (2,046 mm) of rainfall. The snowiest month on record is February 2010, with 36.9 inches (94 cm) of snowfall. The snowiest season (Jul–Jun) on record is 1995–1996, with 75.6 inches (192 cm) of snowfall. The least snowy season was 1972–1973, with 2.3 inches (5.8 cm) of snowfall. The earliest seasonal trace of snowfall occurred on October 10, in both 1979 and 1925. The latest seasonal trace of snowfall occurred on May 9, in both 2020 and 1977.\n\nSee or edit raw graph data.\n\nfile_path: nyc_text.txt\n\n=== Sports ===\n\nNew York City is home to the headquarters of the National Football League, Major League Baseball, the National Basketball Association, the National Hockey League, and Major League Soccer. The New York metropolitan area hosts the most sports teams in the first four major North American professional sports leagues with nine, one more than Los Angeles, and has 11 top-level professional sports teams if Major League Soccer is included, also one more than Los Angeles. Participation in professional sports in the city predates all professional leagues.\nThe city has played host to more than 40 major professional teams in the five sports and their respective competing leagues. Four of the ten most expensive stadiums ever built worldwide (MetLife Stadium, the new Yankee Stadium, Madison Square Garden, and Citi Field) are in the New York metropolitan area. Madison Square Garden, its predecessor, the original Yankee Stadium and Ebbets Field, are sporting venues in New York City, the latter two having been commemorated on U.S. postage stamps. New York was the first of eight American cities to have won titles in all four major leagues (MLB, NHL, NFL and NBA), having done so following the Knicks' 1970 title. In 1972, it became the first city to win titles in five sports when the Cosmos won the NASL final.",
        additional_kwargs={},
    )
    res = _message_to_cohere_documents(message)
    assert len(res) == 5 and all(r["source"] == "nyc_text.txt" for r in res)
