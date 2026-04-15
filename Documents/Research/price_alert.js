// ============================================================
//  FS564 Price Alert Bot
//  Monitors the FS564/fETH price and emails you when it's
//  time to buy or sell.
//
//  SETUP (one time):
//    1. Install Node.js from https://nodejs.org
//    2. Run:  npm install ethers nodemailer
//    3. Fill in YOUR details in the CONFIG section below
//    4. Run:  node price_alert.js
// ============================================================
 
const { ethers } = require("ethers");
const nodemailer = require("nodemailer");
 
// ============================================================
//  CONFIG — fill these in
// ============================================================
const CONFIG = {
  // Your Duke email (where alerts will be sent TO)
  alertEmail: "hb196@duke.edu",
 
  // A Gmail account to SEND from (create a free one just for this)
  // e.g. "fs564alertbot@gmail.com"
  gmailAddress: "tradingblockchain55@gmail.com",
 
  // Gmail App Password (NOT your normal Gmail password)
  // How to get it:
  //   1. Go to myaccount.google.com
  //   2. Search "App Passwords"
  //   3. Create one named "fs564bot"
  //   4. Paste the 16-character code here
  gmailAppPassword: "jihj uihq ucul vzmg",
 
  // RPC endpoint — this lets the script read the blockchain
  // Sign up free at https://www.alchemy.com and paste your key below
  // (make sure to pick the right network your competition uses)
  rpcUrl: "https://eth-sepolia.g.alchemy.com/v2/HbQficraYzwyhFS-N2Xki",
 
  // FS564 token contract address
  fs564Address: "0x3fE9B17A453DE5BCe14F9D782624Fb4919d94553",
 
  // fETH token contract address
  fethAddress:  "0x8D9263E921df92354e5D2dA8F064726887e70756",
 
  // Uniswap V2 pool address for FS564/fETH
  // Find this on Blockscan by searching the FS564 contract
  poolAddress: "0x8A4f3cF94582B96BB80437fA23534C890C03B0F4",
 
  // --- Price thresholds ---
  buyBelow:  0.070,   // Alert to BUY when price drops below this
  sellAbove: 0.095,   // Alert to SELL when price spikes above this
 
  // How often to check (in minutes)
  checkEveryMinutes: 5,
};
 
// ============================================================
//  Uniswap V2 Pair ABI (minimal — just what we need)
// ============================================================
const PAIR_ABI = [
  "function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast)",
  "function token0() external view returns (address)",
];
 
// ============================================================
//  Send email alert
// ============================================================
async function sendEmail(subject, body) {
  const transporter = nodemailer.createTransport({
    service: "gmail",
    auth: {
      user: CONFIG.gmailAddress,
      pass: CONFIG.gmailAppPassword,
    },
  });
 
  await transporter.sendMail({
    from: `"FS564 Alert Bot" <${CONFIG.gmailAddress}>`,
    to: CONFIG.alertEmail,
    subject: subject,
    text: body,
  });
 
  console.log(`📧 Email sent: ${subject}`);
}
 
// ============================================================
//  Fetch current FS564 price in fETH
// ============================================================
async function getPrice() {
  const provider = new ethers.JsonRpcProvider(CONFIG.rpcUrl);
  const pair = new ethers.Contract(CONFIG.poolAddress, PAIR_ABI, provider);
 
  const [reserve0, reserve1] = await pair.getReserves();
  const token0 = await pair.token0();
 
  // Figure out which reserve is FS564 and which is fETH
  const fs564IsToken0 = token0.toLowerCase() === CONFIG.fs564Address.toLowerCase();
 
  const reserveFS564 = fs564IsToken0 ? reserve0 : reserve1;
  const reserveFETH  = fs564IsToken0 ? reserve1 : reserve0;
 
  // Price = fETH per 1 FS564
  const price = Number(reserveFETH) / Number(reserveFS564);
  return price;
}
 
// ============================================================
//  Main loop
// ============================================================
let lastAlertType = null; // Prevents spamming the same alert
 
async function checkPrice() {
  try {
    const price = await getPrice();
    const time = new Date().toLocaleTimeString();
 
    console.log(`[${time}] Current FS564 price: ${price.toFixed(6)} fETH`);
 
    if (price < CONFIG.buyBelow && lastAlertType !== "BUY") {
      lastAlertType = "BUY";
      await sendEmail(
        `🟢 BUY ALERT — FS564 at ${price.toFixed(6)} fETH`,
        `The FS564 price has dropped to ${price.toFixed(6)} fETH, below your buy threshold of ${CONFIG.buyBelow}.\n\nNow is a good time to buy FS564 with your fETH!\n\nGo to: https://tradingcomp.vercel.app/`
      );
    } else if (price > CONFIG.sellAbove && lastAlertType !== "SELL") {
      lastAlertType = "SELL";
      await sendEmail(
        `🔴 SELL ALERT — FS564 at ${price.toFixed(6)} fETH`,
        `The FS564 price has spiked to ${price.toFixed(6)} fETH, above your sell threshold of ${CONFIG.sellAbove}.\n\nConsider selling some FS564 for fETH now!\n\nGo to: https://tradingcomp.vercel.app/`
      );
    } else if (price >= CONFIG.buyBelow && price <= CONFIG.sellAbove) {
      // Price returned to normal range — reset so we can alert again next time
      lastAlertType = null;
    }
 
  } catch (err) {
    console.error("Error checking price:", err.message);
  }
}
 
// ============================================================
//  Start
// ============================================================
console.log("🚀 FS564 Price Alert Bot started!");
console.log(`   Buy alert below:  ${CONFIG.buyBelow} fETH`);
console.log(`   Sell alert above: ${CONFIG.sellAbove} fETH`);
console.log(`   Checking every:   ${CONFIG.checkEveryMinutes} minutes`);
console.log(`   Alerts go to:     ${CONFIG.alertEmail}\n`);
 
checkPrice(); // Run immediately on start
setInterval(checkPrice, CONFIG.checkEveryMinutes * 60 * 1000);
 