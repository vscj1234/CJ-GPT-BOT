const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const mysql = require('mysql2');

const mainUrl = 'https://cloudjune.com';
const outputFilePath = './webcontent.txt';


// MySQL Database Configuration
const dbConfig = {
    host: 'localhost',
    user: 'root',
    password: 'uch$xilC2g',
    database: 'cloudjunebot'
};

const connection = mysql.createConnection(dbConfig);
let allContent = '';
let visitedLinks = new Set();

async function insertIntoDatabase(content) {
    const query = 'INSERT INTO cloudjunebot.webcontent (content) VALUES (?)';
    connection.query(query, [content], (error, results) => {
        if (error) {
            console.error('Error inserting into database:', error);
        } else {
            console.log('Content inserted into database with ID:', results.insertId);
        }
    });
}

async function fetchPageContent(url) {
    try {
        const browser = await puppeteer.launch( {args: ['--no-sandbox'] }
    );
        const page = await browser.newPage();
        
        await page.goto(url, { waitUntil: 'networkidle0' });

        // Exclude header, footer, or other elements by their selectors
        //await page.evaluate(() => {
          //  const header = document.querySelector('header');
            //const footer = document.querySelector('footer');
            
            //if (header) header.remove();
           // if (footer) footer.remove();
            
            // Add additional selectors as needed
       // });

        const textContent = await page.evaluate(() => {
            return document.body.innerText;
        });

        await browser.close();
        return textContent;
    } catch (error) {
        console.error(`Error fetching content from ${url}:`, error.message);
        return ''; // Return empty string for invalid URLs
    }
}

async function fetchAllLinks() {
    const browser = await puppeteer.launch({ args: ['--no-sandbox'] });
    const page = await browser.newPage();

    await page.goto(mainUrl, { waitUntil: 'networkidle0' });

    const links = await page.evaluate(() => {
        const anchorTags = document.querySelectorAll('a');
        return Array.from(anchorTags).map(a => a.href);
    });

    await browser.close();
    return links;
}

async function fetchAndConcatenateContent() {
    const links = await fetchAllLinks();
    for (const link of links) {
        if (!visitedLinks.has(link)) { // Check if link has not been visited
            visitedLinks.add(link);    // Mark link as visited
            console.log(`Fetching content from: ${link}`); // Print the current link
            const content = await fetchPageContent(link);
            if (content) {
                allContent += content + '\n\n';  // Separate content from different pages with two newlines
                //allContent += '-----------------------------------------------------------------\n\n'; 
            }
        }
    }

    await fs.writeFile(outputFilePath, allContent);
    console.log('Content concatenated and saved to', outputFilePath);

    // Insert content into the MySQL database
    insertIntoDatabase(allContent);
}


    connection.query('INSERT INTO webcontent (content) VALUES (?)', [allContent], (error, results) => {
        if (error) throw error;
        console.log('Content saved to MySQL database!');
    });


async function checkForChanges() {
    await fetchAndConcatenateContent();

    // Schedule the next check after 24 hours
    setTimeout(checkForChanges, 86400000); // 24 hours in milliseconds
}

// Start the initial check
checkForChanges().catch(error => {
    console.error('Error:', error);
});
