{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "import requests\n",
    "from bs4 import BeautifulSoup\n",
    "import pandas as pd\n",
    "import re\n",
    "import numpy as np\n",
    "\n",
    "def extract_batting_data(series_id, match_id):\n",
    "\n",
    "    URL = 'https://www.espncricinfo.com/series/'+ str(series_id) + '/scorecard/' + str(match_id)\n",
    "    page = requests.get(URL)\n",
    "    bs = BeautifulSoup(page.content, 'lxml')\n",
    "\n",
    "    table_body=bs.find_all('tbody')\n",
    "    batsmen_df = pd.DataFrame(columns=[\"Name\",\"Desc\",\"Runs\", \"Balls\", \"4s\", \"6s\", \"SR\", \"Team\"])\n",
    "    for i, table in enumerate(table_body[0:4:2]):\n",
    "        rows = table.find_all('tr')\n",
    "        for row in rows[::2]:\n",
    "            cols=row.find_all('td')\n",
    "            cols=[x.text.strip() for x in cols]\n",
    "            if cols[0] == 'Extras':\n",
    "                continue\n",
    "            if len(cols) > 7:\n",
    "                batsmen_df = batsmen_df.append(pd.Series(\n",
    "                [re.sub(r\"\\W+\", ' ', cols[0].split(\"(c)\")[0]).strip(), cols[1], \n",
    "                cols[2], cols[3], cols[5], cols[6], cols[7], i+1], \n",
    "                index=batsmen_df.columns ), ignore_index=True)\n",
    "            else:\n",
    "                batsmen_df = batsmen_df.append(pd.Series(\n",
    "                [re.sub(r\"\\W+\", ' ', cols[0].split(\"(c)\")[0]).strip(), cols[1], \n",
    "                0, 0, 0, 0, 0, i+1], index = batsmen_df.columns), ignore_index=True)\n",
    "                \n",
    "    for i in range(2):\n",
    "        dnb_row = bs.find_all(\"tfoot\")[i].find_all(\"div\")\n",
    "        for c in dnb_row:\n",
    "            dnb_cols = c.find_all('span')\n",
    "            dnb = [x.text.strip().split(\"(c)\")[0] for x in dnb_cols]\n",
    "            dnb = filter(lambda item: item, [re.sub(r\"\\W+\", ' ', x).strip() for x in dnb])\n",
    "            for dnb_batsman in dnb:\n",
    "                batsmen_df = batsmen_df.append(pd.Series([dnb_batsman, \"DNB\", 0, 0, 0, 0, 0, i+1], index = batsmen_df.columns), ignore_index =True)\n",
    "\n",
    "    return batsmen_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_bowling_data(series_id, match_id):\n",
    "\n",
    "    URL = 'https://www.espncricinfo.com/series/'+ str(series_id) + '/scorecard/' + str(match_id)\n",
    "    page = requests.get(URL)\n",
    "    bs = BeautifulSoup(page.content, 'lxml')\n",
    "\n",
    "    table_body=bs.find_all('tbody')\n",
    "    bowler_df = pd.DataFrame(columns=['Name', 'Overs', 'Maidens', 'Runs', 'Wickets',\n",
    "                                      'Econ', 'Dots', '4s', '6s', 'Wd', 'Nb','Team'])\n",
    "    for i, table in enumerate(table_body[1:4:2]):\n",
    "        rows = table.find_all('tr')\n",
    "        for row in rows:\n",
    "            cols=row.find_all('td')\n",
    "            cols=[x.text.strip() for x in cols]\n",
    "            bowler_df = bowler_df.append(pd.Series([cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], \n",
    "                                                    cols[6], cols[7], cols[8], cols[9], cols[10], (i==0)+1], \n",
    "                                                   index=bowler_df.columns ), ignore_index=True)\n",
    "    return bowler_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = extract_batting_data(series_id = 18693, match_id = 1144999)\n",
    "a\n",
    "b = extract_bowling_data(series_id = 18693, match_id = 1144999)\n",
    "b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "with pd.ExcelWriter('output.xlsx') as writer:\n",
    "    a.to_excel(writer, sheet_name='Sheet_name_1')\n",
    "    b.to_excel(writer, sheet_name='Sheet_name_2')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
