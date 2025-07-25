const { test, expect } = require('@playwright/test');
const fs = require('fs').promises;
const http = require('http');

test.describe('Chatbot E2E Tests', () => {
  test.setTimeout(120000); 

  test('Send message and receive bot response', async ({ page }) => {
    const baseUrl = 'http://localhost:3000';
    const testStartTime = new Date().toLocaleString('en-US', { timeZone: 'Africa/Nairobi' });
    console.log(`Test started at: ${testStartTime}`);

   
    const isServerUp = await new Promise(resolve => {
      http
        .get(baseUrl, res => {
          resolve(res.statusCode >= 200 && res.statusCode < 400);
        })
        .on('error', () => resolve(false));
    });
    if (!isServerUp) {
      console.error(`Server at ${baseUrl} is not accessible.`);
      throw new Error('Server is not running or not accessible');
    }

   
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.log(`Browser console error: ${msg.text()}`);
      }
    });

    
    try {
      const response = await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });
      console.log('Navigation status:', response.status());
      expect(response.status()).toBe(200, `Failed to load page: ${baseUrl}`);
      await page.screenshot({ path: `debug-navigation-${Date.now()}.png`, fullPage: true });
    } catch (error) {
      console.error(`Navigation to ${baseUrl} failed:`, error);
      try {
        await page.screenshot({ path: `debug-navigation-failure-${Date.now()}.png`, fullPage: true });
        await fs.writeFile(`debug-navigation-failure-${Date.now()}.html`, await page.content());
      } catch (screenshotError) {
        console.error('Failed to save navigation failure artifacts:', screenshotError);
      }
      throw error;
    }

    
    const currentUrl = page.url();
    console.log('Current URL:', currentUrl);
    if (currentUrl.includes('login')) {
      console.error('Redirected to login page. Authentication may be required.');
      try {
        await page.screenshot({ path: `debug-login-redirect-${Date.now()}.png`, fullPage: true });
        await fs.writeFile(`debug-login-redirect-${Date.now()}.html`, await page.content());
      } catch (screenshotError) {
        console.error('Failed to save login redirect artifacts:', screenshotError);
      }
      throw new Error('Authentication required. Please ensure a pre-authenticated session or revert to login script.');
    }

  
    let html = await page.content();
    await fs.writeFile(`debug-initial-${Date.now()}.html`, html);
    console.log('Initial HTML (first 500 chars):', html.substring(0, 500));

    
    const inputs = await page.evaluate(() => {
      const elements = Array.from(document.querySelectorAll('input, textarea, [contenteditable="true"]'));
      return elements.map(el => ({
        tag: el.tagName,
        id: el.id,
        class: el.className,
        name: el.name,
        type: el.type,
        placeholder: el.placeholder,
      }));
    });
    console.log('Input elements found:', JSON.stringify(inputs, null, 2));

   
    let chatInput = null;
    const possibleInputSelectors = [
      '#chat-input',
      'input[name="chat-input"]',
      '.chat-input',
      'input[type="text"]',
      'textarea.chat-input',
      '#message-input',
      'textarea[name="message"]',
      '[contenteditable="true"]',
      'input[placeholder*="message" i]',
    ];
    for (const selector of possibleInputSelectors) {
      try {
        chatInput = await page.waitForSelector(selector, { state: 'visible', timeout: 10000 });
        console.log(`Found chat input with selector: ${selector}`);
        break;
      } catch (error) {
        console.log(`Selector ${selector} not found, trying next...`);
      }
    }

    if (!chatInput) {
      console.error('No chat input found. Available inputs:', JSON.stringify(inputs, null, 2));
      try {
        await page.screenshot({ path: `debug-input-failure-${Date.now()}.png`, fullPage: true });
        await fs.writeFile(`debug-input-failure-${Date.now()}.html`, await page.content());
      } catch (screenshotError) {
        console.error('Failed to save input failure artifacts:', screenshotError);
      }
      throw new Error('No valid chat input selector found');
    }

    
    const testMessage = 'Hello!';
    await chatInput.type(testMessage);
    try {
      await page.click('button#send-message, button.send-button, button[aria-label*="send" i], button[title*="send" i]', { timeout: 5000 });
      console.log('Message sent using Send button');
    } catch (error) {
      console.log('Send button not found, trying Enter key...');
      await chatInput.press('Enter');
      console.log('Message sent using Enter key');
    }

    
    const messages = await page.evaluate(() => {
      const elements = Array.from(document.querySelectorAll('div, p, span, li, article, section'));
      return elements
        .filter(el => el.textContent.trim().length > 0 && !el.textContent.includes('sveltekit') && !el.textContent.includes('__svelte'))
        .map(el => ({
          tag: el.tagName,
          id: el.id,
          class: el.className,
          text: el.textContent.trim().substring(0, 200),
          dataAttributes: Object.fromEntries(Object.entries(el.dataset)),
        }));
    });
    console.log('Message elements found:', JSON.stringify(messages, null, 2));

   
    const possibleResponseSelectors = [
      '.chat-message.bot',
      '.message.bot-message',
      '.bot-response',
      '.message',
      '[data-role="bot"]',
      'div.bot',
      '.chat-message',
      '.response-text',
      '[data-testid="bot-response"]',
      '.bot-message',
      '.chat-bubble',
      '[data-message-type="bot"]',
      '[data-sveltekit]',
      '.svelte-message',
      'div[class*="svelte"]',
      '[data-svelte-h]',
      '.message-content',
      '.chatbot-response',
    ];
    let botMessageElement = null;
    for (const selector of possibleResponseSelectors) {
      try {
        botMessageElement = await page.waitForSelector(selector, { state: 'visible', timeout: 40000 });
        console.log(`Found bot response with selector: ${selector}`);
        break;
      } catch (error) {
        console.log(`Bot response selector ${selector} not found, trying next...`);
      }
    }

 
    if (!botMessageElement) {
      const responseText = await page.evaluate((testMessage) => {
        const elements = Array.from(document.querySelectorAll('div, p, span, li, article, section'));
        const matchingElement = elements.find(el => {
          const text = el.textContent.toLowerCase().trim();
          return text.length > 0 && (text.includes(testMessage.toLowerCase()) || text.includes('hi') || text.includes('hello') || text.includes('hey')) && !text.includes('sveltekit') && !el.textContent.includes('__svelte');
        });
        return matchingElement ? {
          tag: matchingElement.tagName,
          id: matchingElement.id,
          class: matchingElement.className,
          text: matchingElement.textContent.trim().substring(0, 200),
          dataAttributes: Object.fromEntries(Object.entries(matchingElement.dataset)),
        } : null;
      }, testMessage);
      console.log('Text-based response search:', JSON.stringify(responseText, null, 2));
      if (responseText) {
        const selector = `${responseText.tag}${responseText.id ? `#${responseText.id}` : ''}${responseText.class ? `.${responseText.class.split(' ').join('.')}` : ''}`;
        try {
          botMessageElement = await page.waitForSelector(selector, { state: 'visible', timeout: 10000 });
          console.log(`Found bot response via text search: ${JSON.stringify(responseText)}`);
        } catch (error) {
          console.log(`Text-based selector ${selector} failed:`, error);
        }
      }
    }

    if (!botMessageElement) {
      console.error('No bot response found. Available messages:', JSON.stringify(messages, null, 2));
      try {
        await page.screenshot({ path: `debug-response-failure-${Date.now()}.png`, fullPage: true });
        await fs.writeFile(`debug-response-failure-${Date.now()}.html`, await page.content());
      } catch (screenshotError) {
        console.error('Failed to save response failure artifacts:', screenshotError);
      }
      throw new Error('No valid bot response selector found');
    }

    const botMessage = await botMessageElement.textContent();
    expect(botMessage).toBeTruthy();
    expect(botMessage.trim()).not.toBe('', 'Bot response is empty');
    console.log(`Bot response: ${botMessage}`);

    const testEndTime = new Date().toLocaleString('en-US', { timeZone: 'Africa/Nairobi' });
    console.log(`Test ended at: ${testEndTime}`);
  });
});