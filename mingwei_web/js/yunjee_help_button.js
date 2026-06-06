import { app } from "../../scripts/app.js";

const CONFIG = {
    buttonLabel: "✨ YunJee-ComfyUI ✨",
    introCard: {
        logoUrl: new URL("./icon/1.jpg", import.meta.url).href,
        badge: "AI x Cross-Border Content",
        title: "广州云迹灵动科技有限公司",
        subtitle: "国内领先的「AI + 跨境数字内容全链路服务商」",
        lead: "深耕 AIGC 视觉生成技术的商业落地，以自研 AI 技术为核心、跨境流量运营为抓手，为出海品牌、海外内容平台、跨境电商卖家提供「内容创作 - 流量运营 - 商业变现」的一站式闭环解决方案。",
        tags: ["AIGC视觉生成", "跨境内容服务", "流量运营", "商业变现", "规模化产能"],
        sections: [
            {
                icon: "✦",
                title: "企业定位",
                lines: [
                    "专注于 AI 技术与跨境内容服务深度融合，面向全球化内容业务提供完整解决方案。",
                    "是行业内少有的同时具备 AI 技术研发、跨境本土化内容、流量变现与规模化产能搭建能力的创新型科技企业。"
                ]
            },
            {
                icon: "◆",
                title: "核心驱动",
                lines: [
                    "以自研 AI 技术为底层能力，持续强化 AIGC 视觉生成在商业场景中的落地效率。",
                    "以跨境流量运营为增长抓手，让内容生产、投放与变现形成稳定联动。"
                ]
            },
            {
                icon: "🔗",
                title: "一站式闭环",
                lines: [
                    "为出海品牌、海外内容平台、跨境电商卖家提供「内容创作 - 流量运营 - 商业变现」全链路服务。",
                    "帮助客户从内容生产到商业结果形成更高效、更可复制的增长闭环。"
                ]
            },
            {
                icon: "🚀",
                title: "竞争优势",
                lines: [
                    "兼具技术研发能力与本土化内容理解能力，能够覆盖多语种、多市场、多平台的内容需求。",
                    "同时具备规模化产能搭建与变现能力，适合面向跨境业务的长期增长场景。"
                ]
            }
        ]
    },
    menuItems: [
        {
            label: "🎉 YunJee 介绍",
            action: "show_company_intro",
            title: "🎉 YunJee-ComfyUI 介绍",
            subtitle: "一个打开后就能看到欢迎信息的 ComfyUI 专属 UI 入口。",
            sections: [
                {
                    icon: "💫",
                    title: "风格参考",
                    lines: [
                        "参考了你提供的大炮 ComfyUI 按钮与弹层交互方式。",
                        "保留了顶部入口 + 浮层介绍的体验。"
                    ]
                },
                {
                    icon: "🎨",
                    title: "视觉设计",
                    lines: [
                        "加入了星光、渐变、边框高亮和装饰图标。",
                        "让入口更醒目，也更接近你发来的示意图风格。"
                    ]
                }
            ]
        },
        {
            label: "🚀 ComfyUI镜像包介绍",
            badge: "YunJee-MINGW Mirror Pack",
            title: "🚀 YunJee-ComfyUI 镜像包重磅来袭",
            subtitle: "由 YunJee-MINGW 精心开发，面向高效创作体验打造的开箱即用型 ComfyUI 镜像包。",
            lead: "完美适配 ComfyUI 的 API 调用与常用视频工作流，内置 gemini、nano banana pro、nano banana 2、sora2、veo、grok、seedance2.0 多款实用 API 与多种实用型工作流，开箱即用，轻松解锁高效创作体验。",
            tags: ["ComfyUI API适配", "视频工作流", "Gemini", "Nano Banana Pro", "Sora2", "Veo", "Grok", "Seedance2.0"],
            sections: [
                {
                    icon: "⚙️",
                    title: "核心适配",
                    lines: [
                        "围绕 ComfyUI 的 API 调用做了针对性适配，调用链路更顺手，集成体验更稳定。",
                        "常用视频工作流也已预置到位，减少手动拼装与重复配置成本。"
                    ]
                },
                {
                    icon: "📦",
                    title: "内置能力",
                    lines: [
                        "内置 gemini、nano banana pro、nano banana 2、sora2、veo、grok、seedance2.0 等多款实用 API。",
                        "同时整合多种实用型工作流，适合直接上手进行图像与视频相关创作。"
                    ]
                },
                {
                    icon: "💡",
                    title: "使用体验",
                    lines: [
                        "镜像包强调开箱即用，不需要复杂准备即可进入创作状态。",
                        "更适合希望快速搭建、快速调用、快速出效果的工作场景。"
                    ]
                },
                {
                    icon: "✨",
                    title: "适合人群",
                    lines: [
                        "适合需要频繁使用 API、视频工作流、批量生产内容的创作者与团队。",
                        "也适合希望提升部署效率、降低环境折腾成本的 ComfyUI 使用者。"
                    ]
                }
            ]
        },
        {
            label: "☎ 深度合作需求加VX",
            action: "show_qrcode",
            title: "☎ 深度合作需求加VX",
            subtitle: "扫码添加微信，进一步沟通深度合作需求。",
            qrcodeUrl: new URL("./icon/2.jpg", import.meta.url).href
        }
    ]
};

let globalMenu = null;
let isMenuVisible = false;
let stylesReady = false;

function ensureStyles() {
    if (stylesReady) {
        return;
    }

    const style = document.createElement("style");
    style.textContent = `
        @keyframes yunjeeModalFadeIn {
            from {
                opacity: 0;
                transform: translateY(-18px) scale(0.96);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        @keyframes yunjeeGlow {
            0% {
                box-shadow: 0 0 0 rgba(102, 187, 255, 0);
            }
            50% {
                box-shadow: 0 0 18px rgba(102, 187, 255, 0.22);
            }
            100% {
                box-shadow: 0 0 0 rgba(102, 187, 255, 0);
            }
        }

        @keyframes yunjeeGradientFlow {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }

        @keyframes yunjeePrimaryPulse {
            0% {
                box-shadow: 0 0 0 rgba(68, 145, 255, 0), 0 10px 22px rgba(0, 0, 0, 0.22);
                border-color: rgba(132, 201, 255, 0.34);
            }
            50% {
                box-shadow: 0 0 18px rgba(68, 145, 255, 0.3), 0 0 30px rgba(132, 201, 255, 0.18), 0 10px 22px rgba(0, 0, 0, 0.22);
                border-color: rgba(132, 201, 255, 0.68);
            }
            100% {
                box-shadow: 0 0 0 rgba(68, 145, 255, 0), 0 10px 22px rgba(0, 0, 0, 0.22);
                border-color: rgba(132, 201, 255, 0.34);
            }
        }

        @keyframes yunjeeDangerPulse {
            0% {
                box-shadow: 0 0 0 rgba(255, 95, 95, 0), 0 10px 22px rgba(0, 0, 0, 0.22);
                border-color: rgba(255, 120, 120, 0.38);
            }
            50% {
                box-shadow: 0 0 18px rgba(255, 95, 95, 0.3), 0 0 30px rgba(255, 120, 120, 0.16), 0 10px 22px rgba(0, 0, 0, 0.22);
                border-color: rgba(255, 120, 120, 0.76);
            }
            100% {
                box-shadow: 0 0 0 rgba(255, 95, 95, 0), 0 10px 22px rgba(0, 0, 0, 0.22);
                border-color: rgba(255, 120, 120, 0.38);
            }
        }
    `;
    document.head.appendChild(style);
    stylesReady = true;
}

function buildSection(section) {
    const block = document.createElement("div");
    block.style.cssText = `
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(132, 201, 255, 0.18);
        border-radius: 16px;
        padding: 16px 18px;
    `;

    const heading = document.createElement("div");
    heading.style.cssText = `
        color: #ffffff;
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    `;
    heading.textContent = `${section.icon} ${section.title}`;

    const list = document.createElement("div");
    list.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 8px;
    `;

    section.lines.forEach((line) => {
        const item = document.createElement("div");
        item.style.cssText = `
            color: rgba(255, 255, 255, 0.9);
            font-size: 14px;
            line-height: 1.7;
            padding-left: 10px;
            border-left: 2px solid rgba(132, 201, 255, 0.26);
        `;
        item.textContent = line;
        list.appendChild(item);
    });

    block.appendChild(heading);
    block.appendChild(list);
    return block;
}

function showInfoModal(item) {
    const overlay = document.createElement("div");
    overlay.style.cssText = `
        position: fixed;
        inset: 0;
        z-index: 2147483647;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(8, 12, 22, 0.82);
        backdrop-filter: blur(6px);
        padding: 24px;
    `;

    const modal = document.createElement("div");
    modal.style.cssText = `
        width: min(720px, 92vw);
        max-height: 88vh;
        overflow: auto;
        border-radius: 24px;
        padding: 28px;
        background:
            radial-gradient(circle at top right, rgba(115, 102, 255, 0.24), transparent 32%),
            radial-gradient(circle at top left, rgba(0, 212, 255, 0.18), transparent 26%),
            linear-gradient(160deg, rgba(34, 38, 58, 0.98), rgba(18, 22, 35, 0.98));
        border: 1px solid rgba(132, 201, 255, 0.28);
        box-shadow: 0 22px 60px rgba(0, 0, 0, 0.5);
        animation: yunjeeModalFadeIn 0.24s ease;
    `;

    const headerWrap = document.createElement("div");
    headerWrap.style.cssText = `
        display: flex;
        align-items: center;
        gap: 18px;
        margin-bottom: 18px;
        flex-wrap: wrap;
    `;

    const logoCard = document.createElement("div");
    logoCard.style.cssText = `
        width: 132px;
        min-width: 132px;
        height: 132px;
        border-radius: 24px;
        padding: 12px;
        background:
            radial-gradient(circle at top left, rgba(132, 201, 255, 0.2), transparent 42%),
            linear-gradient(145deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03));
        border: 1px solid rgba(132, 201, 255, 0.18);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06), 0 18px 40px rgba(0, 0, 0, 0.2);
        display: ${item.logoUrl ? "flex" : "none"};
        align-items: center;
        justify-content: center;
        overflow: hidden;
    `;

    const logo = document.createElement("img");
    logo.src = item.logoUrl || "";
    logo.alt = `${item.title || "YunJee"} logo`;
    logo.style.cssText = `
        width: 100%;
        height: 100%;
        object-fit: contain;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.96);
    `;
    logoCard.appendChild(logo);

    const titleWrap = document.createElement("div");
    titleWrap.style.cssText = `
        flex: 1 1 360px;
        min-width: 260px;
    `;

    const badge = document.createElement("div");
    badge.textContent = item.badge || "✨ YunJee Special UI";
    badge.style.cssText = `
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(132, 201, 255, 0.12);
        border: 1px solid rgba(132, 201, 255, 0.22);
        color: #9dd8ff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 14px;
    `;

    const title = document.createElement("div");
    title.textContent = item.title;
    title.style.cssText = `
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
        line-height: 1.25;
        margin-bottom: 10px;
    `;

    const subtitle = document.createElement("div");
    subtitle.textContent = item.subtitle;
    subtitle.style.cssText = `
        color: rgba(255, 255, 255, 0.8);
        font-size: 15px;
        line-height: 1.7;
        margin-bottom: 0;
    `;

    const lead = document.createElement("div");
    lead.textContent = item.lead || "";
    lead.style.cssText = `
        color: rgba(225, 238, 255, 0.92);
        font-size: 14px;
        line-height: 1.85;
        margin-bottom: ${item.lead ? "18px" : "0"};
        padding: ${item.lead ? "14px 16px" : "0"};
        border-radius: 16px;
        background: ${item.lead ? "linear-gradient(135deg, rgba(132, 201, 255, 0.1), rgba(118, 102, 255, 0.08))" : "transparent"};
        border: ${item.lead ? "1px solid rgba(132, 201, 255, 0.16)" : "none"};
    `;

    const tagWrap = document.createElement("div");
    tagWrap.style.cssText = `
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: ${item.tags?.length ? "20px" : "0"};
    `;

    (item.tags || []).forEach((tagText) => {
        const tag = document.createElement("div");
        tag.textContent = tagText;
        tag.style.cssText = `
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(132, 201, 255, 0.1);
            border: 1px solid rgba(132, 201, 255, 0.18);
            color: #bfe6ff;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.2px;
        `;
        tagWrap.appendChild(tag);
    });

    const sectionWrap = document.createElement("div");
    sectionWrap.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 14px;
    `;

    item.sections.forEach((section) => {
        sectionWrap.appendChild(buildSection(section));
    });

    const footer = document.createElement("div");
    footer.style.cssText = `
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        margin-top: 22px;
        flex-wrap: wrap;
    `;

    const tips = document.createElement("div");
    tips.textContent = "🌟 顶部菜单中的 YunJee-ComfyUI 按钮可随时重新打开介绍";
    tips.style.cssText = `
        color: rgba(157, 216, 255, 0.92);
        font-size: 13px;
        line-height: 1.6;
    `;

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "关闭";
    closeBtn.style.cssText = `
        border: 1px solid rgba(132, 201, 255, 0.3);
        background: linear-gradient(135deg, rgba(64, 149, 255, 0.2), rgba(132, 201, 255, 0.12));
        color: #ffffff;
        border-radius: 12px;
        padding: 10px 22px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 700;
        transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
        animation: yunjeeGlow 3s ease-in-out infinite;
    `;

    closeBtn.onmouseenter = () => {
        closeBtn.style.transform = "translateY(-1px)";
        closeBtn.style.borderColor = "rgba(132, 201, 255, 0.55)";
        closeBtn.style.background = "linear-gradient(135deg, rgba(64, 149, 255, 0.34), rgba(132, 201, 255, 0.18))";
    };

    closeBtn.onmouseleave = () => {
        closeBtn.style.transform = "translateY(0)";
        closeBtn.style.borderColor = "rgba(132, 201, 255, 0.3)";
        closeBtn.style.background = "linear-gradient(135deg, rgba(64, 149, 255, 0.2), rgba(132, 201, 255, 0.12))";
    };

    const closeModal = () => {
        document.body.removeChild(overlay);
    };

    closeBtn.onclick = closeModal;
    overlay.onclick = (event) => {
        if (event.target === overlay) {
            closeModal();
        }
    };

    titleWrap.appendChild(badge);
    titleWrap.appendChild(title);
    titleWrap.appendChild(subtitle);
    headerWrap.appendChild(logoCard);
    headerWrap.appendChild(titleWrap);

    modal.appendChild(headerWrap);
    if (item.lead) {
        modal.appendChild(lead);
    }
    if (item.tags?.length) {
        modal.appendChild(tagWrap);
    }
    modal.appendChild(sectionWrap);
    footer.appendChild(tips);
    footer.appendChild(closeBtn);
    modal.appendChild(footer);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
}

function showQrcodeModal(item) {
    const overlay = document.createElement("div");
    overlay.style.cssText = `
        position: fixed;
        inset: 0;
        z-index: 2147483647;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(8, 12, 22, 0.84);
        backdrop-filter: blur(6px);
        padding: 24px;
    `;

    const modal = document.createElement("div");
    modal.style.cssText = `
        width: min(460px, 92vw);
        border-radius: 24px;
        padding: 28px;
        background:
            radial-gradient(circle at top right, rgba(115, 102, 255, 0.22), transparent 34%),
            radial-gradient(circle at top left, rgba(0, 212, 255, 0.16), transparent 28%),
            linear-gradient(160deg, rgba(34, 38, 58, 0.98), rgba(18, 22, 35, 0.98));
        border: 1px solid rgba(132, 201, 255, 0.28);
        box-shadow: 0 22px 60px rgba(0, 0, 0, 0.5);
        text-align: center;
        animation: yunjeeModalFadeIn 0.24s ease;
    `;

    const badge = document.createElement("div");
    badge.textContent = "Deep Cooperation";
    badge.style.cssText = `
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(132, 201, 255, 0.12);
        border: 1px solid rgba(132, 201, 255, 0.22);
        color: #9dd8ff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 14px;
    `;

    const title = document.createElement("div");
    title.textContent = item.title || "☎ 深度合作需求加VX";
    title.style.cssText = `
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
        line-height: 1.25;
        margin-bottom: 10px;
    `;

    const subtitle = document.createElement("div");
    subtitle.textContent = item.subtitle || "扫码添加微信";
    subtitle.style.cssText = `
        color: rgba(255, 255, 255, 0.82);
        font-size: 14px;
        line-height: 1.7;
        margin-bottom: 18px;
    `;

    const qrWrap = document.createElement("div");
    qrWrap.style.cssText = `
        width: 290px;
        height: 290px;
        max-width: 100%;
        margin: 0 auto 18px auto;
        padding: 14px;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.96);
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
    `;

    const qrImage = document.createElement("img");
    qrImage.src = item.qrcodeUrl || "";
    qrImage.alt = "微信二维码";
    qrImage.style.cssText = `
        width: 100%;
        height: 100%;
        object-fit: contain;
        border-radius: 16px;
        display: block;
    `;
    qrWrap.appendChild(qrImage);

    const tips = document.createElement("div");
    tips.textContent = "欢迎扫码联系，沟通深度合作、项目需求与商务对接。";
    tips.style.cssText = `
        color: rgba(157, 216, 255, 0.92);
        font-size: 13px;
        line-height: 1.7;
        margin-bottom: 18px;
    `;

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "关闭";
    closeBtn.style.cssText = `
        border: 1px solid rgba(132, 201, 255, 0.3);
        background: linear-gradient(135deg, rgba(64, 149, 255, 0.2), rgba(132, 201, 255, 0.12));
        color: #ffffff;
        border-radius: 12px;
        padding: 10px 22px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 700;
        transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
        animation: yunjeeGlow 3s ease-in-out infinite;
    `;

    closeBtn.onmouseenter = () => {
        closeBtn.style.transform = "translateY(-1px)";
        closeBtn.style.borderColor = "rgba(132, 201, 255, 0.55)";
        closeBtn.style.background = "linear-gradient(135deg, rgba(64, 149, 255, 0.34), rgba(132, 201, 255, 0.18))";
    };

    closeBtn.onmouseleave = () => {
        closeBtn.style.transform = "translateY(0)";
        closeBtn.style.borderColor = "rgba(132, 201, 255, 0.3)";
        closeBtn.style.background = "linear-gradient(135deg, rgba(64, 149, 255, 0.2), rgba(132, 201, 255, 0.12))";
    };

    const closeModal = () => {
        document.body.removeChild(overlay);
    };

    closeBtn.onclick = closeModal;
    overlay.onclick = (event) => {
        if (event.target === overlay) {
            closeModal();
        }
    };

    modal.appendChild(badge);
    modal.appendChild(title);
    modal.appendChild(subtitle);
    modal.appendChild(qrWrap);
    modal.appendChild(tips);
    modal.appendChild(closeBtn);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
}

function createMenu() {
    if (globalMenu) {
        return globalMenu;
    }

    const menu = document.createElement("div");
    menu.id = "yunjee-comfyui-submenu";
    menu.style.cssText = `
        position: fixed !important;
        min-width: 320px;
        padding: 10px 0;
        display: none;
        z-index: 2147483647 !important;
        border-radius: 16px;
        border: 1px solid rgba(132, 201, 255, 0.28) !important;
        background:
            radial-gradient(circle at top right, rgba(115, 102, 255, 0.2), transparent 30%),
            linear-gradient(180deg, rgba(35, 40, 60, 0.98), rgba(22, 26, 40, 0.98));
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(12px);
        overflow: hidden;
    `;

    CONFIG.menuItems.forEach((item, index) => {
        const menuItem = document.createElement("button");
        menuItem.textContent = item.label;
        menuItem.style.cssText = `
            width: 100%;
            padding: 14px 18px;
            border: none;
            text-align: left;
            background: ${index === 0 ? "rgba(132, 201, 255, 0.06)" : "transparent"};
            color: rgba(255, 255, 255, 0.94);
            cursor: pointer;
            font-size: 14px;
            transition: background 0.18s ease, color 0.18s ease, padding-left 0.18s ease;
            border-top: ${index > 0 ? "1px solid rgba(255, 255, 255, 0.05)" : "none"};
        `;

        menuItem.onmouseenter = () => {
            menuItem.style.background = "rgba(132, 201, 255, 0.14)";
            menuItem.style.color = "#9dd8ff";
            menuItem.style.paddingLeft = "24px";
        };

        menuItem.onmouseleave = () => {
            menuItem.style.background = index === 0 ? "rgba(132, 201, 255, 0.06)" : "transparent";
            menuItem.style.color = "rgba(255, 255, 255, 0.94)";
            menuItem.style.paddingLeft = "18px";
        };

        menuItem.onclick = (event) => {
            event.stopPropagation();
            hideMenu();
            if (item.action === "show_company_intro") {
                showInfoModal(CONFIG.introCard);
            } else if (item.action === "show_qrcode") {
                showQrcodeModal(item);
            } else {
                showInfoModal(item);
            }
        };

        menu.appendChild(menuItem);
    });

    document.body.appendChild(menu);
    globalMenu = menu;
    return menu;
}

function showMenu(buttonRect) {
    const menu = createMenu();
    menu.style.display = "block";
    menu.style.visibility = "hidden";
    menu.style.left = "0px";
    menu.style.top = "0px";

    const menuHeight = menu.offsetHeight || 180;
    const menuWidth = 320;
    let menuLeft = buttonRect.left;
    let menuTop = buttonRect.bottom + 8;

    if (menuTop + menuHeight > window.innerHeight) {
        menuTop = buttonRect.top - menuHeight - 8;
    }

    if (menuTop < 10) {
        menuTop = 10;
    }

    menuLeft = Math.max(10, Math.min(menuLeft, window.innerWidth - menuWidth - 10));

    menu.style.left = `${menuLeft}px`;
    menu.style.top = `${menuTop}px`;
    menu.style.visibility = "visible";
    isMenuVisible = true;
}

function hideMenu() {
    if (!globalMenu) {
        return;
    }

    globalMenu.style.display = "none";
    isMenuVisible = false;
}

function createButton() {
    const button = document.createElement("button");
    button.id = "yunjee-comfyui-button";
    button.textContent = CONFIG.buttonLabel;
    button.style.cssText = `
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        margin: 0 4px;
        padding: 7px 14px;
        border-radius: 12px;
        border: 1px solid rgba(132, 201, 255, 0.34);
        background:
            radial-gradient(circle at top left, rgba(132, 201, 255, 0.14), transparent 40%),
            linear-gradient(120deg, rgba(40, 49, 74, 0.98), rgba(35, 69, 120, 0.96), rgba(26, 31, 49, 0.98));
        background-size: 220% 220%;
        background-position: 0% 50%;
        color: #ffffff;
        font-size: 13px;
        font-weight: 700;
        cursor: pointer;
        white-space: nowrap;
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.22);
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        animation: yunjeeGradientFlow 4.6s ease-in-out infinite, yunjeePrimaryPulse 2.8s ease-in-out infinite;
    `;

    button.onmouseenter = () => {
        button.style.transform = "translateY(-1px)";
        button.style.borderColor = "rgba(132, 201, 255, 0.6)";
        button.style.boxShadow = "0 12px 28px rgba(68, 145, 255, 0.18)";
    };

    button.onmouseleave = () => {
        button.style.transform = "translateY(0)";
        button.style.borderColor = "rgba(132, 201, 255, 0.34)";
        button.style.boxShadow = "0 10px 22px rgba(0, 0, 0, 0.22)";
    };

    button.onclick = (event) => {
        event.stopPropagation();
        if (isMenuVisible) {
            hideMenu();
            return;
        }

        showMenu(button.getBoundingClientRect());
    };

    document.addEventListener("click", (event) => {
        if (isMenuVisible && !button.contains(event.target) && !globalMenu?.contains(event.target)) {
            hideMenu();
        }
    });

    return button;
}

function createCleanButton() {
    const button = document.createElement("button");
    button.id = "yunjee-clean-button";
    button.textContent = "🧹清理";
    button.title = "一键释放所有显存和模型占用";
    button.style.cssText = `
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin: 0 4px;
        padding: 7px 12px;
        border-radius: 12px;
        border: 1px solid rgba(255, 120, 120, 0.38);
        background: linear-gradient(120deg, rgba(92, 42, 42, 0.98), rgba(142, 58, 58, 0.96), rgba(58, 28, 28, 0.98));
        background-size: 220% 220%;
        background-position: 0% 50%;
        color: #ffffff;
        font-size: 13px;
        font-weight: 700;
        cursor: pointer;
        white-space: nowrap;
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.22);
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
        animation: yunjeeGradientFlow 4.2s ease-in-out infinite, yunjeeDangerPulse 2.4s ease-in-out infinite;
    `;

    button.onmouseenter = () => {
        button.style.transform = "translateY(-1px)";
        button.style.borderColor = "rgba(255, 120, 120, 0.7)";
        button.style.background = "linear-gradient(180deg, rgba(118, 48, 48, 0.98), rgba(76, 32, 32, 0.98))";
        button.style.boxShadow = "0 12px 28px rgba(255, 80, 80, 0.18)";
    };

    button.onmouseleave = () => {
        if (button.disabled) {
            return;
        }

        button.style.transform = "translateY(0)";
        button.style.borderColor = "rgba(255, 120, 120, 0.38)";
        button.style.background = "linear-gradient(180deg, rgba(92, 42, 42, 0.98), rgba(58, 28, 28, 0.98))";
        button.style.boxShadow = "0 10px 22px rgba(0, 0, 0, 0.22)";
    };

    button.onclick = async (event) => {
        event.stopPropagation();

        const originalText = button.textContent;
        button.textContent = "🧹清理中...";
        button.disabled = true;
        button.style.cursor = "wait";

        try {
            const response = await fetch("/free", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ unload_models: true, free_memory: true }),
                cache: "no-cache"
            });

            if (response.ok) {
                button.textContent = "✅已释放";
                button.style.background = "linear-gradient(180deg, rgba(38, 100, 62, 0.98), rgba(24, 70, 43, 0.98))";
                button.style.borderColor = "rgba(94, 255, 148, 0.45)";
            } else {
                button.textContent = "❌失败";
            }
        } catch (error) {
            console.error("Free memory request failed:", error);
            button.textContent = "❌错误";
        }

        setTimeout(() => {
            button.textContent = originalText;
            button.disabled = false;
            button.style.cursor = "pointer";
            button.style.transform = "translateY(0)";
            button.style.borderColor = "rgba(255, 120, 120, 0.38)";
            button.style.background = "linear-gradient(180deg, rgba(92, 42, 42, 0.98), rgba(58, 28, 28, 0.98))";
            button.style.boxShadow = "0 10px 22px rgba(0, 0, 0, 0.22)";
        }, 2000);
    };

    return button;
}

function legacyInsert() {
    const menu = document.querySelector(".comfy-menu");
    if (!menu || document.getElementById("yunjee-comfyui-button")) {
        return false;
    }

    const mainButton = createButton();
    const cleanButton = createCleanButton();
    const queueButton = document.getElementById("queue-button");

    if (queueButton) {
        menu.insertBefore(mainButton, queueButton);
        menu.insertBefore(cleanButton, queueButton);
    } else {
        menu.appendChild(mainButton);
        menu.appendChild(cleanButton);
    }

    return true;
}

async function modernInsert() {
    if (document.getElementById("yunjee-comfyui-button")) {
        return true;
    }

    if (app.menu?.actionsGroup?.element) {
        const mainButton = createButton();
        const cleanButton = createCleanButton();
        app.menu.actionsGroup.element.after(mainButton);
        mainButton.after(cleanButton);
        return true;
    }

    return false;
}

app.registerExtension({
    name: "yunjee.helpButton",
    async setup() {
        ensureStyles();

        const inserted = await modernInsert();
        if (!inserted) {
            setTimeout(() => {
                legacyInsert();
            }, 120);
        }
    }
});
